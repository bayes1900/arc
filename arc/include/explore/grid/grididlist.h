#pragma once

#include <base/iarc.h>

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace arc {

class GridIdList {
public:
    using iterator = std::vector<int>::iterator;
    using const_iterator = std::vector<int>::const_iterator;

    void assign(const std::vector<int>& ids) {
        ids_.clear();
        idSet_.clear();
        ids_.reserve(ids.size());
        for (int id : ids) {
            if (idSet_.insert(id).second) {
                ids_.push_back(id);
            }
        }
    }
    bool insert(int id) {
        if (!idSet_.insert(id).second) {
            return false;
        }
        ids_.push_back(id);
        return true;
    }
    bool erase(int id) {
        if (idSet_.erase(id) == 0) {
            return false;
        }
        ids_.erase(std::find(ids_.begin(), ids_.end(), id));
        return true;
    }
    iterator erase(iterator pos) {
        idSet_.erase(*pos);
        return ids_.erase(pos);
    }
    void insertAll(const std::vector<int>& ids) {
        for (int id : ids) {
            insert(id);
        }
    }
    bool contains(int id) const {
        return idSet_.find(id) != idSet_.end();
    }
    void replaceOrder(const std::vector<int>& order) {
        ids_ = order;
        idSet_.clear();
        idSet_.insert(ids_.begin(), ids_.end());
    }
    void clear() {
        ids_.clear();
        idSet_.clear();
    }
    bool empty() const {
        return ids_.empty();
    }
    size_t size() const {
        return ids_.size();
    }
    int front() const {
        return ids_.front();
    }
    int operator[](size_t index) const {
        return ids_[index];
    }
    const std::vector<int>& values() const {
        return ids_;
    }
    iterator begin() {
        return ids_.begin();
    }
    iterator end() {
        return ids_.end();
    }
    const_iterator begin() const {
        return ids_.begin();
    }
    const_iterator end() const {
        return ids_.end();
    }

private:
    std::vector<int> ids_;
    std::unordered_set<int> idSet_;
};

} // arc
