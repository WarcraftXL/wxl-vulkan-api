// Tree wind: the engine's tree models found by their textures, read back once, and drawn through
// their vertex shader's `wind` variant with a consumer's per-vertex data (wxl.vk.render, TREE WIND).
// Copyright (C) 2026 WarcraftXL
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#include "vulkan/Wind.hpp"

#include "shaders/Translated.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Manifest.hpp"
#include "vulkan/Loader.hpp"
#include "vulkan/Memory.hpp"
#include "wxl/Script.hpp"
#include "wxl/offsets/engine/Camera.hpp"
#include "wxl/offsets/engine/Gx.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    namespace context = wxlvk::vk::context;
    namespace memory  = wxlvk::vk::memory;
    namespace gx      = wxl::offsets::engine::gx;
    namespace gxdev   = wxl::offsets::engine::gxdevice;
    using wxlvk::draw::Cmd;
    using wxlvk::vk::wind::Draw;

    constexpr uint32_t     kSlots         = 2;
    constexpr uint32_t     kMaxModels     = 4096;        ///< every model ever keyed (a stale key takes a new one)
    constexpr uint32_t     kMaxBatches    = 64;
    constexpr uint32_t     kMaxRereads    = 3;           ///< read again when a section is met after the read
    constexpr uint64_t     kFreeAfter     = 4;           ///< frames before a freed data range is reused (in flight)
    constexpr uint64_t     kSettleFrames  = 90;          ///< frames without a new batch before the read-back
    constexpr VkDeviceSize kReadbackBytes = 4u << 20;    ///< one model's indices, then its vertex range
    constexpr VkDeviceSize kDataBytes     = 16u << 20;   ///< every windy model's vec4s: 1 048 576 vertices (the expansions' big trees are 15-25k each)
    constexpr VkDeviceSize kBlockBytes    = 4608;        ///< WXL_VkWindParams, rounded
    static_assert(sizeof(WXL_VkWindParams) <= kBlockBytes, "the wind block holds WXL_VkWindParams");
    constexpr uint32_t     kMaxInstances  = 2048;        ///< windy tree instances listed a frame

    enum class State : uint8_t { Collecting, Reading, Ready, Handed, Windy, Rejected };

    struct Batch
    {
        uint32_t firstIndex, indexCount, alphaTested, blend;
        uint16_t name;
        uint32_t start;            ///< firstIndex less its index buf's base: the same for every instance's copy
        VkBuffer ib;               ///< the index pool it was seen in (instances' bufs may sit in others)
        uint32_t indexPoolBytes;
    };

    struct Model
    {
        VkBuffer vb = VK_NULL_HANDLE;
        uintptr_t bufId = 0;       ///< the engine's vertex buf (Cmd::vertexBufId): a key reused by another one is stale
        uint32_t rereads = 0;
        bool     late = false;     ///< a section was met after the batches were read
        VkDeviceSize dataAt = 0, dataBytes = 0;   ///< its per-vertex data's range (draw.on)
        int32_t  vertexOffset = 0;
        uint32_t stride = 0;
        char     vs[Cmd::kNameChars] = {};
        State    state = State::Collecting;
        Batch    batches[kMaxBatches] = {};
        uint32_t batchCount = 0;
        uint64_t changed = 0;      ///< the frame a batch was last added
        uint64_t seen = 0;         ///< the frame a draw last named it
        uint32_t items = 0;        ///< the vertex buf's vertex count, the largest a draw reported
        uint32_t bufBytes = 0;     ///< the vertex buf's bytes (bounds the whole-buf read-back)
        uint32_t poolBytes = 0;    ///< the vertex pool's, the source buffer's size (bounds every vertex copy)
        uint32_t indexPoolBytes = 0;
        // The read-back mesh (Ready / Handed), released once answered.
        std::vector<uint8_t>         vertices;
        std::vector<uint16_t>        indices;
        std::vector<WXL_VkWindBatch> out;
        uint32_t minIndex = 0, vertexCount = 0;
        Draw     draw = {};
    };

    struct Job
    {
        bool     active = false;
        uint32_t model = 0;
        int      phase = 0;        ///< 1 the indices, 2 the vertex range
        uint64_t serial = 0;
        uint32_t indexCount = 0;   ///< phase 1's, in order of the batches
    };

    struct TexEntry
    {
        uintptr_t user = 0, fill = 0;
        int32_t   name = -1;       ///< into names, -1 none
        bool      watched = false;
        bool      unit = false;    ///< a character's, a creature's or an item's: the draw's model is a unit's
    };

    struct Wind
    {
        bool watching = false;
        std::vector<std::string> patterns;   ///< lowercase, '/' as '\'
        std::unordered_map<uintptr_t, TexEntry> textures;
        std::unordered_map<uint64_t, uint64_t> unitModels;   ///< model key -> its buffer id: the models a unit's texture was drawn with
        std::vector<std::string> names;
        std::unordered_map<std::string, uint16_t> nameIndex;

        std::vector<Model> models;
        std::unordered_map<uint64_t, uint32_t> modelAt;
        /// The last key Observe() resolved and its model: a model's batches come in runs, and the hashed
        /// lookup was most of a watched draw's cost. Only Observe() changes modelAt, and it drops this on an erase.
        uint64_t lastKey = 0;
        uint32_t lastModel = 0;
        bool     lastValid = false;
        /// MoverDraw's previous inputs and answer.
        struct MoverMemo
        {
            bool        valid = false;
            VkBuffer    vb    = VK_NULL_HANDLE;
            int32_t     vertexOffset = 0;
            uint32_t    stride = 0;
            uint64_t    bufId = 0;
            const void* tex  = nullptr;
            uintptr_t   user = 0, fill = 0;
            bool        mover = false;
        } mover;
        uint64_t frame = 0;
        Job      job;

        bool ready = false, failed = false;
        memory::Buffer readback, data, blocks[kSlots];
        VkDeviceSize   dataUsed = 0;
        struct FreeRange { VkDeviceSize at, bytes; uint64_t frame; };
        std::vector<FreeRange> freeRanges;   ///< data ranges given back, reused once no frame in flight reads them
        uint32_t stale = 0;                  ///< keys found reused by another vertex buf
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE, emptyLayout = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet  sets[kSlots] = {};

        WXL_VkWindParams params = {};
        bool             paramsSet = false;
        uint64_t         viewFrame = ~0ull;   ///< the frame whose live camera the slot's block holds
        std::vector<WXL_VkWindInstance> instances;   ///< this frame's

        std::vector<std::string> noVariant;
        uint32_t drawnNow = 0, fellNow = 0, drawnLast = 0, fellLast = 0;   ///< this frame's, the last full frame's
        const char* fellWhy = "";
        struct Stats { uint64_t observed = 0, windy = 0, readbacks = 0, rejected = 0, windyModels = 0, full = 0; } stats;
    };
    Wind g;

    std::string Lower(const char* s)
    {
        std::string out(s ? s : "");
        for (char& ch : out) ch = ch == '/' ? '\\' : static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return out;
    }

    /// A model is its vertices: the pool buffer, the place in it and the stride. Not the index buffer --
    /// every instance of a model indexes the same vertices from its own index buf -- and checked against
    /// the engine's vertex buf on every draw (Observe), since a freed model's place is handed to another.
    uint64_t KeyOf(VkBuffer vb, int32_t offset, uint32_t stride)
    {
        uint64_t h = 1469598103934665603ull;
        // A non-dispatchable handle is a 64-bit integer on a 32-bit build, a pointer on a 64-bit one.
        const uint64_t parts[3] = { (uint64_t)(vb), static_cast<uint32_t>(offset), stride };
        for (uint64_t p : parts) h = (h ^ p) * 1099511628211ull;
        return h;
    }

    /// The engine texture's file path, and whether a pattern matches it (cached per CGxTex, checked
    /// against its user argument and fill so a reused CGxTex is looked at again).
    const TexEntry& Texture(const void* tex)
    {
        static const TexEntry kNone;
        if (!tex) return kNone;
        const uint8_t* t = static_cast<const uint8_t*>(tex);
        const uintptr_t user = *reinterpret_cast<const uintptr_t*>(t + gxdev::kTexUserArg);
        const uintptr_t fill = *reinterpret_cast<const uintptr_t*>(t + gxdev::kTexFillCallback);
        TexEntry& e = g.textures[reinterpret_cast<uintptr_t>(tex)];
        if (e.user == user && e.fill == fill && (e.name >= 0 || e.user != 0)) return e;
        e = TexEntry();
        e.user = user;
        e.fill = fill;
        // Gx.hpp: only a file-loaded texture's fill is handed its CTexture, which holds the name.
        if ((fill == gx::kTexFillBlp || fill == gx::kTexFillBlob || fill == gx::kTexFillTga) && user
            && *reinterpret_cast<const uint32_t*>(user) == gx::kTexHandleVTable)
        {
            const char* name = reinterpret_cast<const char*>(user + gx::kTexHandleNameField);
            char bounded[0x104];
            std::strncpy(bounded, name, sizeof bounded - 1);
            bounded[sizeof bounded - 1] = '\0';
            const std::string low = Lower(bounded);
            for (const std::string& p : g.patterns)
                if (low.find(p) != std::string::npos) e.watched = true;
            // The path's first folder tells a unit's model (a character, a creature, an item it carries)
            // from the world's: the shadow caches' movers (pipelines/Record.cpp).
            e.unit = low.rfind("character\\", 0) == 0 || low.rfind("creature\\", 0) == 0 || low.rfind("item\\", 0) == 0;
            auto it = g.nameIndex.find(low);
            if (it == g.nameIndex.end() && g.names.size() < 0xFFFF)
            {
                it = g.nameIndex.emplace(low, static_cast<uint16_t>(g.names.size())).first;
                g.names.push_back(bounded);
            }
            if (it != g.nameIndex.end()) e.name = it->second;
        }
        return e;
    }

    void Barrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkAccessFlags srcAccess, VkPipelineStageFlags dst,
                 VkAccessFlags dstAccess)
    {
        VkMemoryBarrier m = {};
        m.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        m.srcAccessMask = srcAccess;
        m.dstAccessMask = dstAccess;
        context::Get()->dapi->CmdPipelineBarrier(cmd, src, dst, 0, 1, &m, 0, nullptr, 0, nullptr);
    }

    bool Create()
    {
        const context::Handles* h = context::Get();
        if (!h || !h->dapi) return false;
        const auto& d = *h->dapi;
        g.failed = true;   // until the end
        if (!memory::CreateBuffer(kReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, memory::Where::HostVisible, g.readback)
            || !g.readback.block.mapped
            || !memory::CreateBuffer(kDataBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memory::Where::HostVisible, g.data)
            || !g.data.block.mapped)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: wind: no host-visible memory for the read-back and data buffers: "
                                "no tree wind this session");
            return false;
        }
        for (memory::Buffer& b : g.blocks)
            if (!memory::CreateBuffer(kBlockBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, memory::Where::Stream, b) || !b.block.mapped)
                return false;

        VkDescriptorSetLayoutBinding bind[2] = {};
        bind[0].binding         = 0;
        bind[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bind[0].descriptorCount = 1;
        bind[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        bind[1].binding         = 1;
        bind[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bind[1].descriptorCount = 1;
        bind[1].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 2;
        li.pBindings    = bind;
        if (d.CreateDescriptorSetLayout(h->device, &li, nullptr, &g.setLayout) != VK_SUCCESS) return false;
        VkDescriptorSetLayoutCreateInfo empty = {};
        empty.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        if (d.CreateDescriptorSetLayout(h->device, &empty, nullptr, &g.emptyLayout) != VK_SUCCESS) return false;

        const VkDescriptorPoolSize sizes[2] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kSlots },
                                                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kSlots } };
        VkDescriptorPoolCreateInfo pi = {};
        pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets       = kSlots;
        pi.poolSizeCount = 2;
        pi.pPoolSizes    = sizes;
        if (d.CreateDescriptorPool(h->device, &pi, nullptr, &g.pool) != VK_SUCCESS) return false;
        const VkDescriptorSetLayout layouts[kSlots] = { g.setLayout, g.setLayout };
        VkDescriptorSetAllocateInfo ai = {};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = g.pool;
        ai.descriptorSetCount = kSlots;
        ai.pSetLayouts        = layouts;
        if (d.AllocateDescriptorSets(h->device, &ai, g.sets) != VK_SUCCESS) return false;
        for (uint32_t i = 0; i < kSlots; ++i)
        {
            const VkDescriptorBufferInfo ub = { g.blocks[i].handle, 0, kBlockBytes };
            const VkDescriptorBufferInfo sb = { g.data.handle, 0, kDataBytes };
            VkWriteDescriptorSet w[2] = {};
            for (uint32_t k = 0; k < 2; ++k)
            {
                w[k].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[k].dstSet          = g.sets[i];
                w[k].dstBinding      = k;
                w[k].descriptorCount = 1;
                w[k].descriptorType  = k == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[k].pBufferInfo     = k == 0 ? &ub : &sb;
            }
            d.UpdateDescriptorSets(h->device, 2, w, 0, nullptr);
            std::memset(g.blocks[i].block.mapped, 0, kBlockBytes);
            memory::FlushHost(g.blocks[i].block);
        }
        g.failed = false;
        g.ready  = true;
        wxl::ScriptMgr::Log(WXL_LOG_INFO, "vulkan: wind: ready: %u KiB of per-vertex data (%u vertices), a %u KiB read-back; "
                            "watching %u texture pattern(s)", static_cast<unsigned>(kDataBytes >> 10),
                            static_cast<unsigned>(kDataBytes / 16), static_cast<unsigned>(kReadbackBytes >> 10),
                            static_cast<unsigned>(g.patterns.size()));
        return true;
    }

    /// A 4 x 4 inverse (Gauss-Jordan); false when singular.
    bool Inverse(const float m[16], float out[16])
    {
        float a[4][8];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 8; ++c) a[r][c] = c < 4 ? m[r * 4 + c] : (c - 4 == r ? 1.0f : 0.0f);
        for (int c = 0; c < 4; ++c)
        {
            int piv = c;
            for (int r = c + 1; r < 4; ++r)
                if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
            if (std::fabs(a[piv][c]) < 1e-12f) return false;
            if (piv != c)
                for (int k = 0; k < 8; ++k) std::swap(a[c][k], a[piv][k]);
            const float inv = 1.0f / a[c][c];
            for (int k = 0; k < 8; ++k) a[c][k] *= inv;
            for (int r = 0; r < 4; ++r)
                if (r != c)
                {
                    const float f = a[r][c];
                    for (int k = 0; k < 8; ++k) a[r][k] -= f * a[c][k];
                }
        }
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) out[r * 4 + c] = a[r][c + 4];
        return true;
    }

    /// The slot's block, with this frame's live camera: the trees' draws carry this frame's model -> view
    /// rows, and a view a frame older (the consumer's, set after the opaque world) put every tree off
    /// by the eye's travel since -- a shake as fast as the frames when the eye flies.
    void RefreshView(uint32_t frameSlot)
    {
        g.viewFrame = g.frame;
        if (frameSlot >= kSlots || !g.ready) return;
        float view[16];
        std::memcpy(view, reinterpret_cast<const float*>(wxl::offsets::engine::camera::kView), sizeof view);
        bool identity = true;
        for (int i = 0; i < 16; ++i) identity = identity && view[i] == ((i % 5 == 0) ? 1.0f : 0.0f);
        float inv[16];
        if (identity || !Inverse(view, inv)) return;   // no world camera now: the consumer's view stays
        std::memcpy(g.params.view, view, sizeof view);
        std::memcpy(g.params.viewInv, inv, sizeof inv);
        std::memcpy(g.blocks[frameSlot].block.mapped, &g.params, sizeof g.params);
        memory::FlushHost(g.blocks[frameSlot].block);
    }

    /// Gives a model's data range back (reused kFreeAfter frames on) and stops its wind.
    void FreeData(Model& m)
    {
        if (m.dataBytes) g.freeRanges.push_back({ m.dataAt, m.dataBytes, g.frame });
        m.dataAt = m.dataBytes = 0;
        m.draw = Draw();
    }

    /// A data range: a freed one no frame in flight still reads (first fit), else past the last.
    bool AllocData(VkDeviceSize bytes, VkDeviceSize& at)
    {
        for (size_t i = 0; i < g.freeRanges.size(); ++i)
        {
            Wind::FreeRange& r = g.freeRanges[i];
            if (r.bytes < bytes || g.frame < r.frame + kFreeAfter) continue;
            at = r.at;
            r.at += bytes;
            r.bytes -= bytes;
            if (!r.bytes) g.freeRanges.erase(g.freeRanges.begin() + static_cast<std::ptrdiff_t>(i));
            return true;
        }
        if (g.dataUsed + bytes > kDataBytes) return false;
        at = g.dataUsed;
        g.dataUsed += bytes;
        return true;
    }

    void Reject(Model& m, const char* why)
    {
        FreeData(m);
        m.state = State::Rejected;
        m.vertices.clear();
        m.vertices.shrink_to_fit();
        m.indices.clear();
        m.indices.shrink_to_fit();
        m.out.clear();
        ++g.stats.rejected;
        if (why)
            wxl::ScriptMgr::Log(WXL_LOG_INFO, "vulkan: wind: model %u (%s, %u batch(es)) left unwound: %s",
                                static_cast<unsigned>(&m - g.models.data()), m.vs, m.batchCount, why);
    }

    /// Phase 1 done: the indices are in the read-back. Their range, and phase 2's copy.
    void StartVertices(VkCommandBuffer cmd, uint64_t serial)
    {
        Model& m = g.models[g.job.model];
        const uint16_t* idx = static_cast<const uint16_t*>(g.readback.block.mapped);
        uint32_t lo = 0xFFFF, hi = 0;
        for (uint32_t i = 0; i < g.job.indexCount; ++i)
        {
            lo = std::min<uint32_t>(lo, idx[i]);
            hi = std::max<uint32_t>(hi, idx[i]);
        }
        if (lo > hi)
        {
            g.job.active = false;
            Reject(m, "no indices");
            return;
        }
        // The whole buf (every vertex of the model, whatever batches were seen), when its size is
        // known and the indices stay inside it.
        // Never past the buf's own bytes, nor the pool buffer the copy reads.
        uint32_t items = m.items;
        if (m.bufBytes && m.stride && VkDeviceSize(items) * m.stride > m.bufBytes) items = m.bufBytes / m.stride;
        if (items > hi && VkDeviceSize(items) * m.stride <= kReadbackBytes / 2
            && (!m.poolBytes || (VkDeviceSize(m.vertexOffset) + items) * m.stride <= m.poolBytes))
        {
            lo = 0;
            hi = items - 1;
        }
        const uint32_t count = hi - lo + 1;
        const VkDeviceSize bytes = VkDeviceSize(count) * m.stride;
        const VkDeviceSize indexBytes = VkDeviceSize(g.job.indexCount) * 2;
        const VkDeviceSize at = (indexBytes + 15) & ~VkDeviceSize(15);
        if (m.vertexOffset < 0 || at + bytes > kReadbackBytes)
        {
            g.job.active = false;
            Reject(m, "its vertex range does not fit the read-back");
            return;
        }
        if (!m.poolBytes || (VkDeviceSize(m.vertexOffset) + lo) * m.stride + bytes > m.poolBytes)
        {
            g.job.active = false;
            Reject(m, "its vertex range is past its pool buffer (or the pool's size is unknown)");
            return;
        }
        m.indices.assign(idx, idx + g.job.indexCount);
        for (uint16_t& v : m.indices) v = static_cast<uint16_t>(v - lo);
        m.minIndex    = lo;
        m.vertexCount = count;
        VkBufferCopy copy = {};
        copy.srcOffset = (VkDeviceSize(m.vertexOffset) + lo) * m.stride;
        copy.dstOffset = at;
        copy.size      = bytes;
        Barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT);
        context::Get()->dapi->CmdCopyBuffer(cmd, m.vb, g.readback.handle, 1, &copy);
        Barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                VK_ACCESS_HOST_READ_BIT);
        g.job.phase  = 2;
        g.job.serial = serial;
    }

    /// Phase 2 done: the mesh, for the consumer.
    void FinishModel()
    {
        Model& m = g.models[g.job.model];
        const VkDeviceSize at = (VkDeviceSize(g.job.indexCount) * 2 + 15) & ~VkDeviceSize(15);
        const uint8_t* src = static_cast<const uint8_t*>(g.readback.block.mapped) + at;
        m.vertices.assign(src, src + size_t(m.vertexCount) * m.stride);
        m.out.clear();
        uint32_t first = 0;
        for (uint32_t b = 0; b < m.batchCount; ++b)
        {
            const Batch& bt = m.batches[b];
            WXL_VkWindBatch o = {};
            o.firstIndex  = first;
            o.indexCount  = bt.indexCount;
            o.alphaTested = bt.alphaTested;
            o.blend       = bt.blend;
            o.texture     = bt.name != 0xFFFF && bt.name < g.names.size() ? g.names[bt.name].c_str() : "";
            m.out.push_back(o);
            first += bt.indexCount;
        }
        m.state = State::Ready;
        ++g.stats.readbacks;
        g.job.active = false;
    }
}

namespace wxlvk::vk::wind
{
    bool UnitTexture(const void* tex) { return Texture(tex).unit; }
    bool UnnamedTexture(const void* tex) { return tex != nullptr && Texture(tex).name < 0; }

    bool MoverDraw(const Cmd& c)
    {
        const void* tex  = c.engineTex0;
        uintptr_t   user = 0, fill = 0;
        if (tex)
        {
            const uint8_t* t = static_cast<const uint8_t*>(tex);
            user = *reinterpret_cast<const uintptr_t*>(t + gxdev::kTexUserArg);
            fill = *reinterpret_cast<const uintptr_t*>(t + gxdev::kTexFillCallback);
        }
        Wind::MoverMemo& m = g.mover;
        if (m.valid && m.vb == c.vertexBuffer && m.vertexOffset == c.vertexOffset && m.stride == c.vertexStride
            && m.bufId == static_cast<uint64_t>(c.vertexBufId) && m.tex == tex && m.user == user && m.fill == fill)
            return m.mover;
        const bool mover = UnitModel(c) || UnnamedTexture(tex);
        m.valid        = true;
        m.vb           = c.vertexBuffer;
        m.vertexOffset = c.vertexOffset;
        m.stride       = c.vertexStride;
        m.bufId        = static_cast<uint64_t>(c.vertexBufId);
        m.tex          = tex;
        m.user         = user;
        m.fill         = fill;
        m.mover        = mover;
        return mover;
    }

    bool UnitModel(const Cmd& c)
    {
        if (c.vertexBuffer == VK_NULL_HANDLE) return Texture(c.engineTex0).unit;
        // The model, keyed as the wind keys it; a place the engine reused for another model is forgotten.
        const uint64_t key = KeyOf(c.vertexBuffer, c.vertexOffset, c.vertexStride);
        auto it = g.unitModels.find(key);
        if (it != g.unitModels.end() && it->second != static_cast<uint64_t>(c.vertexBufId))
        {
            g.unitModels.erase(it);
            it = g.unitModels.end();
        }
        if (it != g.unitModels.end()) return true;
        if (!Texture(c.engineTex0).unit) return false;
        if (g.unitModels.size() < 4096) g.unitModels.emplace(key, static_cast<uint64_t>(c.vertexBufId));
        return true;
    }

    const Draw* Observe(const Cmd& c, uint32_t frameSlot)
    {
        if (!g.watching) return nullptr;
        if (g.paramsSet && g.params.time[2] < 0.5f) return nullptr;   // the consumer turned it off
        // The engine's M2 draws: an indexed triangle list through a Diffuse_* vertex shader.
        if (!c.indexed || c.topology != 3 || std::strncmp(c.vs, "Diffuse_", 8) != 0 || c.vertexStride < 12
            || c.vertexBuffer == VK_NULL_HANDLE || c.indexBuffer == VK_NULL_HANDLE)
            return nullptr;
        ++g.stats.observed;
        const uint64_t key = KeyOf(c.vertexBuffer, c.vertexOffset, c.vertexStride);
        constexpr uint32_t kNone = 0xFFFFFFFFu;
        uint32_t at = kNone;
        if (g.lastValid && g.lastKey == key)
            at = g.lastModel;
        else
        {
            const auto found = g.modelAt.find(key);
            if (found != g.modelAt.end()) at = found->second;
        }
        // The engine freed that model and put another one at its place: the key is the new one's.
        if (at != kNone)
        {
            Model& old = g.models[at];
            if (old.bufId != c.vertexBufId || old.items != c.vertexItems || old.bufBytes != c.vertexBufBytes)
            {
                if (old.state != State::Reading && !(g.job.active && g.job.model == at))
                {
                    FreeData(old);
                    old.state = State::Rejected;
                    old.vertices.clear();
                    old.indices.clear();
                    old.out.clear();
                    ++g.stale;
                    g.modelAt.erase(key);
                    g.lastValid = false;
                    at = kNone;
                }
                else
                    return nullptr;   // its read is in flight: the new model is keyed once it lands
            }
        }
        if (at == kNone)
        {
            if (!Texture(c.engineTex0).watched || g.models.size() >= kMaxModels) return nullptr;
            // Only a model whose vertex shader has a wind variant can sway: the others (a module that
            // blends bones) are never read back, and their draws never take a wind key.
            const wxlvk::vk::manifest::Record* vr = wxlvk::shaders::translated::Resolve(c.vs, false);
            const bool variant = vr && (vr->wind[0] != '\0' || vr->windBones[0] != '\0');
            Model m;
            m.vb = c.vertexBuffer;
            m.bufId = c.vertexBufId;
            m.items = c.vertexItems;
            m.bufBytes = c.vertexBufBytes;
            m.vertexOffset = c.vertexOffset;
            m.stride = c.vertexStride;
            std::strncpy(m.vs, c.vs, sizeof m.vs - 1);
            m.changed = g.frame;
            g.models.push_back(std::move(m));
            at = static_cast<uint32_t>(g.models.size() - 1);
            g.modelAt.emplace(key, at);
            g.lastKey   = key;
            g.lastModel = at;
            g.lastValid = true;
            if (!variant)
            {
                NoVariant(c.vs);
                Reject(g.models.back(), nullptr);
                return nullptr;
            }
        }
        g.lastKey   = key;
        g.lastModel = at;
        g.lastValid = true;
        Model& m = g.models[at];
        m.seen = g.frame;
        if (c.vertexPoolBytes) m.poolBytes = m.poolBytes ? std::min(m.poolBytes, c.vertexPoolBytes) : c.vertexPoolBytes;
        if (m.state == State::Rejected) return nullptr;
        // Its sections, whichever instance draws them: an instance's copy of one (its own index buf,
        // the same start in it and the same count) is the same section. One met after the read --
        // culled or hidden while the model was collected -- has the model read again (its wind kept
        // meanwhile), so no section is left out of the analysis for good.
        {
            const uint32_t start = c.firstIndex - std::min(c.firstIndex, c.indexBase);
            bool known = false;
            for (uint32_t b = 0; b < m.batchCount && !known; ++b)
                known = m.batches[b].indexCount == c.indexCount
                        && (m.batches[b].firstIndex == c.firstIndex || m.batches[b].start == start);
            if (!known && m.batchCount < kMaxBatches)
            {
                const TexEntry& t = Texture(c.engineTex0);
                Batch& b = m.batches[m.batchCount++];
                b.firstIndex  = c.firstIndex;
                b.indexCount  = c.indexCount;
                b.alphaTested = (c.alphaRef > 0 || c.blend == 1) ? 1u : 0u;
                b.blend       = c.blend;
                b.name        = t.name >= 0 ? static_cast<uint16_t>(t.name) : 0xFFFF;
                b.start       = start;
                b.ib          = c.indexBuffer;
                b.indexPoolBytes = c.indexPoolBytes;
                m.changed     = g.frame;
                if (m.state != State::Collecting && m.rereads < kMaxRereads) m.late = true;
            }
            if (m.late && m.state == State::Windy)
            {
                m.late = false;
                ++m.rereads;
                m.state = State::Collecting;   // m.draw stays on: the old data until the new is set
            }
        }
        if (!m.draw.on || !g.paramsSet || !g.ready) return nullptr;
        ++g.stats.windy;
        if (g.viewFrame != g.frame) RefreshView(frameSlot);
        // The instance, once however many batches it draws: its model -> view rows. A static model's
        // are c31..c33; a skinned one's are its bones' (c31 + 3 i), and bone 0 is often a hidden one,
        // scaled to nothing -- its rows put every point of the model on its pivot. So the first bone
        // whose rows are a real transform (a scale between 1/50 and 50, a non-flat 3 x 3).
        int bone = -1;
        if (c.vertexRows)
            for (int b = 0; b < 11 && bone < 0; ++b)
            {
                const float* r0 = c.vertexRows[31 + b * 3];
                const float* r1 = c.vertexRows[32 + b * 3];
                const float* r2 = c.vertexRows[33 + b * 3];
                const float s2 = r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2];
                const float det = r0[0] * (r1[1] * r2[2] - r1[2] * r2[1]) - r0[1] * (r1[0] * r2[2] - r1[2] * r2[0]) +
                                  r0[2] * (r1[0] * r2[1] - r1[1] * r2[0]);
                if (s2 > 4e-4f && s2 < 2500.0f && std::fabs(det) > 1e-6f) bone = b;
            }
        if (bone >= 0 && g.instances.size() < kMaxInstances)
        {
            WXL_VkWindInstance inst = {};
            inst.model = at;
            inst.bone = static_cast<uint32_t>(bone);
            for (int r = 0; r < 3; ++r)
                for (int k = 0; k < 4; ++k) inst.rows[r * 4 + k] = c.vertexRows[31 + bone * 3 + r][k];
            bool same = false;
            for (size_t j = g.instances.size(); j-- > 0 && j + 8 >= g.instances.size();)
                if (g.instances[j].model == inst.model && std::memcmp(g.instances[j].rows, inst.rows, sizeof inst.rows) == 0)
                {
                    same = true;
                    break;
                }
            if (!same) g.instances.push_back(inst);
        }
        return &m.draw;
    }

    void BeginFrame(uint32_t frameSlot)
    {
        if (!g.watching) return;
        ++g.frame;
        g.drawnLast = g.drawnNow;
        g.fellLast  = g.fellNow;
        g.drawnNow  = g.fellNow = 0;
        g.instances.clear();
        if (!g.ready && !g.failed && !Create())
        {
            g.failed = true;
            return;
        }
        if (!g.ready || frameSlot >= kSlots) return;
        std::memcpy(g.blocks[frameSlot].block.mapped, &g.params, sizeof g.params);
        memory::FlushHost(g.blocks[frameSlot].block);
    }

    VkDescriptorSetLayout SetLayout() { return g.ready ? g.setLayout : VK_NULL_HANDLE; }
    VkDescriptorSetLayout EmptyLayout() { return g.ready ? g.emptyLayout : VK_NULL_HANDLE; }
    VkDescriptorSet       Set(uint32_t frameSlot) { return g.ready && frameSlot < kSlots ? g.sets[frameSlot] : VK_NULL_HANDLE; }
    bool                  Ready() { return g.ready; }

    bool PrePassWanted()
    {
        if (!g.ready || !g.watching) return false;
        if (g.paramsSet && g.params.time[2] < 0.5f && !g.job.active) return false;   // off: no new read-back
        if (g.job.active) return true;
        for (const Model& m : g.models)
            if (m.state == State::Collecting && m.batchCount && g.frame - m.changed > kSettleFrames) return true;
        return false;
    }

    void PrePass(VkCommandBuffer cmd, uint64_t serial, uint64_t completed)
    {
        if (!g.ready || !cmd) return;
        if (g.job.active)
        {
            if (completed < g.job.serial) return;
            if (g.job.phase == 1)
            {
                StartVertices(cmd, serial);
                return;
            }
            FinishModel();
        }
        // The next settled model, still drawn (its pool buffer alive): its indices.
        for (uint32_t i = 0; i < g.models.size(); ++i)
        {
            Model& m = g.models[i];
            if (m.state != State::Collecting || !m.batchCount || g.frame - m.changed <= kSettleFrames) continue;
            if (g.frame - m.seen > 1) continue;   // not drawn last frame: try again when it is
            uint32_t total = 0;
            bool inside = true;
            for (uint32_t b = 0; b < m.batchCount; ++b)
            {
                total += m.batches[b].indexCount;
                if (!m.batches[b].indexPoolBytes || m.batches[b].ib == VK_NULL_HANDLE
                    || (VkDeviceSize(m.batches[b].firstIndex) + m.batches[b].indexCount) * 2 > m.batches[b].indexPoolBytes)
                    inside = false;
            }
            if (!inside)
            {
                Reject(m, "a batch's indices are past its index pool (or the pool's size is unknown)");
                continue;
            }
            if (VkDeviceSize(total) * 2 > kReadbackBytes / 2)
            {
                Reject(m, "too many indices for the read-back");
                continue;
            }
            VkDeviceSize at = 0;
            Barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT);
            for (uint32_t b = 0; b < m.batchCount; ++b)
            {
                VkBufferCopy copy = {};
                copy.srcOffset = VkDeviceSize(m.batches[b].firstIndex) * 2;
                copy.dstOffset = at;
                copy.size      = VkDeviceSize(m.batches[b].indexCount) * 2;
                at += copy.size;
                context::Get()->dapi->CmdCopyBuffer(cmd, m.batches[b].ib, g.readback.handle, 1, &copy);
            }
            Barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                    VK_ACCESS_HOST_READ_BIT);
            m.state = State::Reading;
            g.job = Job();
            g.job.active = true;
            g.job.model = i;
            g.job.phase = 1;
            g.job.serial = serial;
            g.job.indexCount = total;
            return;
        }
    }

    void Watch(const char* const* substrings, uint32_t count)
    {
        g.models.reserve(kMaxModels);   // TakeModel's pointers survive new models
        g.patterns.clear();
        for (uint32_t i = 0; substrings && i < count; ++i)
            if (substrings[i] && substrings[i][0]) g.patterns.push_back(Lower(substrings[i]));
        g.textures.clear();   // looked at again against the new list
        g.mover.valid = false;
        g.watching = !g.patterns.empty();
        wxl::ScriptMgr::Log(WXL_LOG_INFO, "vulkan: wind: watching %u texture pattern(s) for tree models",
                            static_cast<unsigned>(g.patterns.size()));
    }

    int TakeModel(WXL_VkWindModel* out)
    {
        if (!out || out->structSize < sizeof(uint32_t)) return 0;
        for (uint32_t i = 0; i < g.models.size(); ++i)
        {
            Model& m = g.models[i];
            if (m.state != State::Ready) continue;
            m.state = State::Handed;
            WXL_VkWindModel o = {};
            o.structSize   = out->structSize;
            o.id           = i;
            o.vertexCount  = m.vertexCount;
            o.vertexStride = m.stride;
            o.vertices     = m.vertices.data();
            o.indexCount   = static_cast<uint32_t>(m.indices.size());
            o.indices      = m.indices.data();
            o.batchCount   = static_cast<uint32_t>(m.out.size());
            o.batches      = m.out.data();
            o.vertexShader = m.vs;
            std::memcpy(out, &o, std::min<size_t>(out->structSize, sizeof o));
            return 1;
        }
        return 0;
    }

    int SetModel(uint32_t id, const float* perVertex, uint32_t vertexCount, uint32_t vec4sPerVertex)
    {
        if (vec4sPerVertex != 1 && vec4sPerVertex != 4) return 0;
        if (id >= g.models.size()) return 0;
        Model& m = g.models[id];
        if (m.state != State::Handed) return 0;
        if (!perVertex || vertexCount != m.vertexCount)
        {
            Reject(m, nullptr);
            return perVertex ? 0 : 1;
        }
        const VkDeviceSize bytes = VkDeviceSize(vertexCount) * 16 * vec4sPerVertex;
        VkDeviceSize at = 0;
        if (!g.ready || vertexCount == 0 || !AllocData(bytes, at))
        {
            ++g.stats.full;
            Reject(m, "the wind data buffer is full");
            return 0;
        }
        // A model read again: its new data in a new range, the old one freed once no frame reads it.
        FreeData(m);
        std::memcpy(static_cast<uint8_t*>(g.data.block.mapped) + at, perVertex, size_t(bytes));
        memory::FlushHost(g.data.block);
        m.dataAt     = at;
        m.dataBytes  = bytes;
        m.draw.bias  = m.vertexOffset + static_cast<int32_t>(m.minIndex);
        m.draw.start = static_cast<uint32_t>(at / 16);
        m.draw.count = vertexCount;
        m.draw.on    = vec4sPerVertex;
        m.state      = State::Windy;
        if (m.late && m.rereads < kMaxRereads)
        {
            m.late = false;
            ++m.rereads;
            m.state = State::Collecting;
        }
        m.vertices.clear();
        m.vertices.shrink_to_fit();
        m.indices.clear();
        m.indices.shrink_to_fit();
        m.out.clear();
        ++g.stats.windyModels;
        return 1;
    }

    void SetParams(const WXL_VkWindParams* params)
    {
        if (!params || params->structSize < sizeof(uint32_t)) return;
        WXL_VkWindParams p = {};
        p.look[0] = p.look[1] = p.look[2] = p.look[3] = 1.0f;   // an older consumer's struct stops short of them
        std::memcpy(&p, params, std::min<size_t>(params->structSize, sizeof p));
        p.structSize = sizeof p;
        g.params    = p;
        g.paramsSet = true;
    }

    void Drawn() { ++g.drawnNow; }
    void FellBack(const char* why)
    {
        ++g.fellNow;
        g.fellWhy = why ? why : "";
    }

    uint32_t Instances(WXL_VkWindInstance* out, uint32_t max)
    {
        const uint32_t n = std::min<uint32_t>(max, static_cast<uint32_t>(g.instances.size()));
        if (out && n) std::memcpy(out, g.instances.data(), sizeof(WXL_VkWindInstance) * n);
        return n;
    }

    int Stats(WXL_VkWindStats* out)
    {
        if (!out || out->structSize < sizeof(uint32_t)) return 0;
        WXL_VkWindStats o = {};
        o.structSize = out->structSize;
        for (const Model& m : g.models)
        {
            if (m.state == State::Collecting || m.state == State::Reading) ++o.collecting;
            if (m.state == State::Windy) ++o.windyModels;
            if (m.state == State::Rejected) ++o.rejected;
        }
        o.models       = static_cast<uint32_t>(g.models.size());
        o.windyDraws   = g.drawnLast;
        o.fallbacks    = g.fellLast;
        o.fallbackWhy  = g.fellWhy;
        o.noVariant    = static_cast<uint32_t>(g.noVariant.size());
        o.ready        = g.ready ? 1u : 0u;
        o.paramsSet    = g.paramsSet ? 1u : 0u;
        o.dataKiB      = static_cast<uint32_t>(g.dataUsed >> 10);
        std::memcpy(o.time, g.params.time, sizeof o.time);
        std::memcpy(o.wind, g.params.wind, sizeof o.wind);
        std::memcpy(out, &o, std::min<size_t>(out->structSize, sizeof o));
        return 1;
    }

    void NoVariant(const char* vertexShader)
    {
        const std::string name = vertexShader ? vertexShader : "";
        if (std::find(g.noVariant.begin(), g.noVariant.end(), name) != g.noVariant.end()) return;
        g.noVariant.push_back(name);
        wxl::ScriptMgr::Log(WXL_LOG_INFO, "vulkan: wind: '%s' has no wind variant (skinned or not translated): its "
                            "trees are drawn still", name.c_str());
    }

    void LogStats()
    {
        if (!g.watching) return;
        uint32_t collecting = 0, windy = 0, rereads = 0;
        for (const Model& m : g.models)
        {
            if (m.state == State::Collecting) ++collecting;
            if (m.state == State::Windy) ++windy;
            rereads += m.rereads;
        }
        VkDeviceSize freed = 0;
        for (const Wind::FreeRange& r : g.freeRanges) freed += r.bytes;
        wxl::ScriptMgr::Log(WXL_LOG_INFO, "vulkan: wind: %u stale key(s) re-keyed (a model's place reused by another), "
                            "%u read(s) again for a section met late, %u KiB of data freed for reuse", g.stale, rereads,
                            static_cast<unsigned>(freed >> 10));
        wxl::ScriptMgr::Log(WXL_LOG_INFO, "vulkan: wind: %u model(s) watched (%u collecting, %u windy, %llu rejected), "
                            "%llu read back; %llu M2 draw(s) observed, %llu windy; last frame %u drawn with a wind "
                            "pipeline, %u fell back (%s); data %u of %u KiB; wind %.2f, strength %.2f, on %.0f",
                            static_cast<unsigned>(g.models.size()), collecting, windy,
                            static_cast<unsigned long long>(g.stats.rejected), static_cast<unsigned long long>(g.stats.readbacks),
                            static_cast<unsigned long long>(g.stats.observed), static_cast<unsigned long long>(g.stats.windy),
                            g.drawnLast, g.fellLast, g.fellWhy[0] ? g.fellWhy : "none",
                            static_cast<unsigned>(g.dataUsed >> 10), static_cast<unsigned>(kDataBytes >> 10),
                            g.params.wind[2], g.params.time[1], g.params.time[2]);
    }
}
