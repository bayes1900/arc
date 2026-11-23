#pragma once

#include <base/iarc.h>
#include <env/sdfmap.h>

namespace arc {

struct MultiMapManagerParams {
    int droneId{1};
    int droneNum{2};
    size_t chunkSize{200};
    double minStampInterval{0.3};
};

struct MapChunk {
    uint32_t idx{0};
    std::vector<size_t> voxelAdrs;
    std::vector<uint8_t> voxelOcc;
    bool needQuery{false};
    bool empty{true};
};

struct ChunksData {
    std::vector<MapChunk> chunks;
    std::vector<int> idxList;
};

struct ChunkStamps {
    int fromDroneId{0};
    std::vector<std::vector<int>> idxLists;
    double time{0.0};
};

struct ChunkData {
    int fromDroneId{0};
    int toDroneId{0};
    int chunkDroneId{0};
    std::vector<size_t> voxelAdrs;
    std::vector<uint8_t> voxelOcc;
    uint32_t idx{0};
};

struct ChunksBox {
    Eigen::Vector3d min{Eigen::Vector3d::Zero()};
    Eigen::Vector3d max{Eigen::Vector3d::Zero()};
    bool valid{false};
};

class MultiMapManager {
public:
    MultiMapManager(SdfMap& map, const MultiMapManagerParams& params)
        : map_(map)
        , params_(params)
    {
        if (params_.droneNum < 1 || params_.droneId < 1 || params_.droneId > params_.droneNum) {
            throw std::invalid_argument("invalid drone id / drone number");
        }
        params_.chunkSize = std::max<size_t>(1, params_.chunkSize);
        multiMapChunks_.resize(params_.droneNum);
        chunkBuffer_.resize(params_.droneNum);
        bufferMap_.resize(params_.droneNum);
        lastChunkStampTime_.assign(params_.droneNum, 0.0);
        chunkBoxes_.resize(params_.droneNum);
    }
    void updateMapChunk(const std::vector<size_t>& adrs) {
        adrBuffer_.insert(adrBuffer_.end(), adrs.begin(), adrs.end());
        while (adrBuffer_.size() > params_.chunkSize) {
            MapChunk chunk;
            chunk.voxelAdrs.assign(adrBuffer_.begin(), adrBuffer_.begin() + params_.chunkSize);
            chunk.idx = static_cast<uint32_t>(multiMapChunks_[params_.droneId - 1].chunks.size() + 1);
            chunk.needQuery = true;
            chunk.empty = false;
            multiMapChunks_[params_.droneId - 1].chunks.push_back(std::move(chunk));
            adrBuffer_.erase(adrBuffer_.begin(), adrBuffer_.begin() + params_.chunkSize);

            std::vector<int>& idxList = multiMapChunks_[params_.droneId - 1].idxList;
            if (idxList.empty()) {
                idxList.push_back(1);
                idxList.push_back(static_cast<int>(multiMapChunks_[params_.droneId - 1].chunks.back().idx));
            } else {
                idxList.back() = static_cast<int>(multiMapChunks_[params_.droneId - 1].chunks.back().idx);
            }
        }
    }
    ChunkStamps buildStamp(double time) const {
        ChunkStamps stamp;
        stamp.fromDroneId = params_.droneId;
        stamp.time = time;
        stamp.idxLists.resize(params_.droneNum);
        for (int i = 0; i < params_.droneNum; ++ i) {
            stamp.idxLists[i] = multiMapChunks_[i].idxList;
        }
        return stamp;
    }
    std::vector<ChunkData> handleStamp(const ChunkStamps& stamp) {
        std::vector<ChunkData> outgoing;
        if (stamp.fromDroneId == params_.droneId) {
            return outgoing;
        }
        if (stamp.fromDroneId < 1 || stamp.fromDroneId > params_.droneNum) {
            return outgoing;
        }
        if (static_cast<int>(stamp.idxLists.size()) < params_.droneNum) {
            return outgoing;
        }
        if (stamp.time > 1e-9) {
            const double lastTime = lastChunkStampTime_[stamp.fromDroneId - 1];
            if (stamp.time - lastTime < params_.minStampInterval) {
                return outgoing;
            }
            lastChunkStampTime_[stamp.fromDroneId - 1] = stamp.time;
        }
        for (int i = 0; i < params_.droneNum; ++ i) {
            if (i == stamp.fromDroneId - 1) {
                continue;
            }
            std::vector<int> missed;
            findMissedChunkIds(multiMapChunks_[i].idxList, stamp.idxLists[i], missed);
            collectChunks_(i + 1, stamp.fromDroneId, missed, outgoing);
        }
        return outgoing;
    }
    void receiveChunk(const ChunkData& chunk) {
        if (chunk.fromDroneId == params_.droneId) {
            return;
        }
        if (chunk.chunkDroneId < 1 || chunk.chunkDroneId > params_.droneNum || chunk.idx == 0 ||
            chunk.voxelAdrs.empty()) {
            return;
        }
        auto& received = bufferMap_[chunk.chunkDroneId - 1];
        const int key = static_cast<int>(chunk.idx);
        if (received.find(key) != received.end()) {
            return;
        }
        chunkBuffer_[chunk.chunkDroneId - 1].push_back(chunk);
        received[key] = 1;
    }
    void processReceivedChunks() {
        for (int i = 0; i < params_.droneNum; ++ i) {
            auto& buffer = chunkBuffer_[i];
            if (buffer.empty()) {
                continue;
            }
            std::sort(buffer.begin(), buffer.end(), [](const ChunkData& lhs, const ChunkData& rhs) {
                return lhs.idx < rhs.idx;
            });
            std::vector<int> idxList;
            int lastIdx = static_cast<int>(buffer.front().idx);
            idxList.push_back(lastIdx);
            for (size_t j = 1; j < buffer.size(); ++ j) {
                const int currentIdx = static_cast<int>(buffer[j].idx);
                if (currentIdx - lastIdx > 1) {
                    idxList.push_back(lastIdx);
                    idxList.push_back(currentIdx);
                }
                lastIdx = currentIdx;
            }
            idxList.push_back(lastIdx);

            ChunksData& chunksData = multiMapChunks_[i];
            const int existingSize = static_cast<int>(chunksData.chunks.size());
            if (lastIdx > existingSize) {
                chunksData.chunks.resize(static_cast<size_t>(lastIdx));
                for (int j = existingSize; j < lastIdx; ++ j) {
                    chunksData.chunks[j].idx = static_cast<uint32_t>(j + 1);
                    chunksData.chunks[j].empty = true;
                    chunksData.chunks[j].needQuery = false;
                }
            }
            for (const ChunkData& data : buffer) {
                if (data.idx == 0 || data.idx > chunksData.chunks.size()) {
                    continue;
                }
                MapChunk& chunk = chunksData.chunks[data.idx - 1];
                if (!chunk.empty) {
                    continue;
                }
                chunk.voxelAdrs = data.voxelAdrs;
                chunk.voxelOcc = data.voxelOcc;
                chunk.idx = data.idx;
                chunk.empty = false;
                chunk.needQuery = false;
                applyChunkToMap_(chunk, data.chunkDroneId);
            }
            std::vector<int> merged;
            mergeChunkIds(idxList, chunksData.idxList, merged);
            chunksData.idxList = std::move(merged);
            buffer.clear();
            bufferMap_[i].clear();
        }
    }
    void getChunkBoxes(
        std::vector<Eigen::Vector3d>& mins,
        std::vector<Eigen::Vector3d>& maxs,
        bool reset = true
    ) {
        for (ChunksBox& box : chunkBoxes_) {
            if (!box.valid) {
                continue;
            }
            mins.push_back(box.min);
            maxs.push_back(box.max);
            if (reset) {
                box.valid = false;
            }
        }
    }
    static void findMissedChunkIds(
        const std::vector<int>& self,
        const std::vector<int>& other,
        std::vector<int>& missed
    ) {
        missed.clear();
        if (self.empty()) {
            return;
        }
        if (other.empty()) {
            missed = self;
            return;
        }
        const auto selfIntervals = toIntervals_(self);
        const auto otherIntervals = toIntervals_(other);
        for (const auto& interval : selfIntervals) {
            int current = interval.first;
            for (const auto& covered : otherIntervals) {
                if (covered.second < current) {
                    continue;
                }
                if (covered.first > interval.second) {
                    break;
                }
                if (covered.first > current) {
                    missed.push_back(current);
                    missed.push_back(covered.first - 1);
                }
                current = std::max(current, covered.second + 1);
                if (current > interval.second) {
                    break;
                }
            }
            if (current <= interval.second) {
                missed.push_back(current);
                missed.push_back(interval.second);
            }
        }
    }
    static void mergeChunkIds(
        const std::vector<int>& list1,
        const std::vector<int>& list2,
        std::vector<int>& output
    ) {
        std::vector<std::pair<int, int>> intervals = toIntervals_(list1);
        const auto second = toIntervals_(list2);
        intervals.insert(intervals.end(), second.begin(), second.end());
        std::sort(intervals.begin(), intervals.end());
        output.clear();
        for (const auto& interval : intervals) {
            if (output.empty() || interval.first > output.back() + 1) {
                output.push_back(interval.first);
                output.push_back(interval.second);
            } else {
                output.back() = std::max(output.back(), interval.second);
            }
        }
    }
    const std::vector<ChunksData>& chunks() const {
        return multiMapChunks_;
    }
    const std::vector<ChunksBox>& chunkBoxes() const {
        return chunkBoxes_;
    }

private:
    static std::vector<std::pair<int, int>> toIntervals_(const std::vector<int>& list) {
        std::vector<std::pair<int, int>> intervals;
        for (size_t i = 0; i + 1 < list.size(); i += 2) {
            if (list[i] <= list[i + 1]) {
                intervals.emplace_back(list[i], list[i + 1]);
            }
        }
        std::sort(intervals.begin(), intervals.end());
        std::vector<std::pair<int, int>> merged;
        for (const auto& interval : intervals) {
            if (merged.empty() || interval.first > merged.back().second + 1) {
                merged.push_back(interval);
            } else {
                merged.back().second = std::max(merged.back().second, interval.second);
            }
        }
        return merged;
    }
    void collectChunks_(
        int chunkDroneId,
        int toDroneId,
        const std::vector<int>& idxList,
        std::vector<ChunkData>& outgoing
    ) {
        if (chunkDroneId < 1 || chunkDroneId > params_.droneNum) {
            return;
        }
        ChunksData& data = multiMapChunks_[chunkDroneId - 1];
        for (size_t i = 0; i + 1 < idxList.size(); i += 2) {
            for (int idx = idxList[i]; idx <= idxList[i + 1]; ++ idx) {
                if (idx < 1 || idx > static_cast<int>(data.chunks.size())) {
                    continue;
                }
                MapChunk& chunk = data.chunks[idx - 1];
                if (chunk.empty) {
                    continue;
                }
                if (chunk.needQuery && chunkDroneId == params_.droneId) {
                    fillOcc_(chunk);
                }
                ChunkData dataMsg;
                dataMsg.fromDroneId = params_.droneId;
                dataMsg.toDroneId = toDroneId;
                dataMsg.chunkDroneId = chunkDroneId;
                dataMsg.voxelAdrs = chunk.voxelAdrs;
                dataMsg.voxelOcc = chunk.voxelOcc;
                dataMsg.idx = chunk.idx;
                outgoing.push_back(std::move(dataMsg));
            }
        }
    }
    void fillOcc_(MapChunk& chunk) {
        chunk.voxelOcc.clear();
        chunk.voxelOcc.reserve(chunk.voxelAdrs.size());
        const MapData& md = map_.data();
        const MapParams& mp = map_.params();
        for (size_t adr : chunk.voxelAdrs) {
            if (adr >= md.numVoxels) {
                chunk.voxelOcc.push_back(0);
                continue;
            }
            chunk.voxelOcc.push_back(md.occupancyLogits[adr] > mp.locc ? 1 : 0);
        }
        chunk.needQuery = false;
    }
    void applyChunkToMap_(const MapChunk& chunk, int chunkDroneId) {
        map_.applyExternalVoxels(chunk.voxelAdrs, chunk.voxelOcc);
        ChunksBox& box = chunkBoxes_[chunkDroneId - 1];
        const MapData& md = map_.data();
        const MapParams& mp = map_.params();
        for (size_t adr : chunk.voxelAdrs) {
            if (adr >= md.numVoxels) {
                continue;
            }
            const Eigen::Vector3d pos = mp.i2p(mp.voxelIdxToIndex(adr));
            if (!box.valid) {
                box.min = pos;
                box.max = pos;
                box.valid = true;
            } else {
                box.min = box.min.cwiseMin(pos);
                box.max = box.max.cwiseMax(pos);
            }
        }
    }

private:
    SdfMap& map_;
    MultiMapManagerParams params_;
    std::vector<ChunksData> multiMapChunks_;
    std::vector<size_t> adrBuffer_;
    std::vector<std::vector<ChunkData>> chunkBuffer_;
    std::vector<std::unordered_map<int, char>> bufferMap_;
    std::vector<double> lastChunkStampTime_;
    std::vector<ChunksBox> chunkBoxes_;
};

} // arc
