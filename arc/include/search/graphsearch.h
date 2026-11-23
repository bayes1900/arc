#pragma once

#include <base/iarc.h>

namespace arc {

template <typename NodeT>
class GraphSearch {
public:
    GraphSearch() = default;
    ~GraphSearch() = default;

public:
    void addNode(NodeT* node) {
        if (!node) {
            return;
        }
        nodes_.push_back(node);
    }
    void addEdge(int from, int to) {
        if (from < 0 || static_cast<size_t>(from) >= nodeNum() || to < 0 || static_cast<size_t>(to) >= nodeNum()) {
            return;
        }
        nodes_[from]->neighbors.push_back(nodes_[to]);
        edgeNum_ += 1;
    }
    bool dijkstraSearch(int start, int goal, std::vector<NodeT*>& path) {
        path.clear();
        if (start < 0 || static_cast<size_t>(start) >= nodeNum() || goal < 0 || static_cast<size_t>(goal) >= nodeNum()) {
            return false;
        }
        states_.clear();
        states_.reserve(nodes_.size());
        for (NodeT* node : nodes_) {
            states_.emplace(node, SearchState{});
        }
        using OpenEntry = std::pair<double, NodeT*>;
        struct OpenEntryCompare {
            bool operator()(const OpenEntry& lhs, const OpenEntry& rhs) const {
                return lhs.first > rhs.first;
            }
        };
        std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenEntryCompare> openSet;

        NodeT* startNode = nodes_[start];
        NodeT* goalNode = nodes_[goal];
        states_.at(startNode).gValue = 0.0;
        openSet.emplace(0.0, startNode);

        while (!openSet.empty()) {
            const auto [queuedGValue, current] = openSet.top();
            openSet.pop();

            auto& currentState = states_.at(current);
            if (currentState.closed || queuedGValue > currentState.gValue) {
                continue;
            }
            currentState.closed = true;

            if (current == goalNode) {
                NodeT* node = current;
                while (node) {
                    path.push_back(node);
                    node = states_.at(node).parent;
                }
                std::reverse(path.begin(), path.end());
                return true;
            }
            for (NodeT* neighbor : current->neighbors) {
                auto& neighborState = states_.at(neighbor);
                if (neighborState.closed) {
                    continue;
                }
                const double candidate = currentState.gValue + current->costTo(*neighbor);
                if (candidate < neighborState.gValue) {
                    neighborState.gValue = candidate;
                    neighborState.parent = current;
                    openSet.emplace(candidate, neighbor);
                }
            }
        }
        return false;
    }
    double gValue(const NodeT* node) const {
        if (!node) {
            return std::numeric_limits<double>::max() / 4.0;
        }
        const auto iter = states_.find(node);
        if (iter == states_.end()) {
            return std::numeric_limits<double>::max() / 4.0;
        }
        return iter->second.gValue;
    }
    size_t nodeNum() const {
        return nodes_.size();
    }
    size_t edgeNum() const {
        return edgeNum_;
    }

private:
    struct SearchState {
        double gValue{std::numeric_limits<double>::max() / 4.0};
        bool closed{false};
        NodeT* parent{nullptr};
    };

private:
    std::vector<NodeT*> nodes_;
    std::unordered_map<const NodeT*, SearchState> states_;
    size_t edgeNum_{0};
};

} // arc
