#pragma once

// Incremental kd-tree, adapted from HKU-MARS ikd-Tree.

#include <base/base.h>

#include <pthread.h>
#include <unistd.h>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif

namespace arc {

struct BoxPointType {
    float vertex_min[3];
    float vertex_max[3];
};

enum class DeleteStorage {
    NOT_RECORD,
    DELETE_POINTS_REC,
    MULTI_THREAD_REC,
};

template <typename PointType>
class KdTree {
public:
    static constexpr double EPS = 1e-6;
    static constexpr int MINIMAL_UNBALANCED_TREE_SIZE = 10;
    static constexpr int MULTI_THREAD_REBUILD_POINT_NUM = 1500;
    static constexpr bool DOWNSAMPLE_SWITCH = true;

    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    using Ptr = std::shared_ptr<KdTree<PointType>>;

    enum class Operation {
        ADD_POINT,
        DELETE_POINT,
        DELETE_BOX,
        ADD_BOX,
        DOWNSAMPLE_DELETE,
        PUSH_DOWN,
    };

    struct KdTreeNode {
        PointType point;
        int division_axis;
        int TreeSize = 1;
        int invalid_point_num = 0;
        int down_del_num = 0;
        bool point_deleted = false;
        bool tree_deleted = false;
        bool point_downsample_deleted = false;
        bool tree_downsample_deleted = false;
        bool need_push_down_to_left = false;
        bool need_push_down_to_right = false;
        bool working_flag = false;
        pthread_mutex_t push_down_mutex_lock;
        float node_range_x[2], node_range_y[2], node_range_z[2];
        float radius_sq;
        KdTreeNode* left_son_ptr = nullptr;
        KdTreeNode* right_son_ptr = nullptr;
        KdTreeNode* father_ptr = nullptr;
        float alpha_del;
        float alpha_bal;
    };

    struct OperationLog {
        PointType point;
        BoxPointType boxpoint;
        bool tree_deleted;
        bool tree_downsample_deleted;
        Operation op;
    };

    struct PointTypeCmp {
        PointType point;
        float dist = 0.0f;

        PointTypeCmp(PointType p = PointType(), float d = INFINITY) {
            point = p;
            dist = d;
        }

        bool operator<(const PointTypeCmp& a) const {
            if (std::fabs(dist - a.dist) < 1e-10f) {
                return point.x < a.point.x;
            }
            return dist < a.dist;
        }
    };

    class ManualHeap {
    public:
        explicit ManualHeap(int maxCapacity = 100)
            : heap_(new PointTypeCmp[maxCapacity])
            , heapSize_(0)
            , cap_(maxCapacity) {
        }

        ~ManualHeap() {
            delete[] heap_;
        }

        void pop() {
            if (heapSize_ == 0) {
                return;
            }
            heap_[0] = heap_[heapSize_ - 1];
            heapSize_ --;
            moveDown_(0);
        }

        PointTypeCmp top() {
            return heap_[0];
        }

        void push(PointTypeCmp point) {
            if (heapSize_ >= cap_) {
                return;
            }
            heap_[heapSize_] = point;
            floatUp_(heapSize_);
            heapSize_ ++;
        }

        int size() {
            return heapSize_;
        }

        void clear() {
            heapSize_ = 0;
        }

    private:
        void moveDown_(int heapIndex) {
            int l = heapIndex * 2 + 1;
            PointTypeCmp tmp = heap_[heapIndex];
            while (l < heapSize_) {
                if (l + 1 < heapSize_ && heap_[l] < heap_[l + 1]) {
                    l ++;
                }
                if (tmp < heap_[l]) {
                    heap_[heapIndex] = heap_[l];
                    heapIndex = l;
                    l = heapIndex * 2 + 1;
                } else {
                    break;
                }
            }
            heap_[heapIndex] = tmp;
        }

        void floatUp_(int heapIndex) {
            int ancestor = (heapIndex - 1) / 2;
            PointTypeCmp tmp = heap_[heapIndex];
            while (heapIndex > 0) {
                if (heap_[ancestor] < tmp) {
                    heap_[heapIndex] = heap_[ancestor];
                    heapIndex = ancestor;
                    ancestor = (heapIndex - 1) / 2;
                } else {
                    break;
                }
            }
            heap_[heapIndex] = tmp;
        }

        PointTypeCmp* heap_;
        int heapSize_;
        int cap_;
    };

    class ManualQueue {
    public:
        static constexpr int CAPACITY = 1000000;

        void pop() {
            if (counter_ == 0) {
                return;
            }
            head_ ++;
            head_ %= CAPACITY;
            counter_ --;
            if (counter_ == 0) {
                isEmpty_ = true;
            }
        }

        OperationLog front() {
            return q_[head_];
        }

        OperationLog back() {
            return q_[tail_];
        }

        void clear() {
            head_ = 0;
            tail_ = 0;
            counter_ = 0;
            isEmpty_ = true;
        }

        void push(OperationLog op) {
            q_[tail_] = op;
            counter_ ++;
            if (isEmpty_) {
                isEmpty_ = false;
            }
            tail_ ++;
            tail_ %= CAPACITY;
        }

        bool empty() {
            return isEmpty_;
        }

        int size() {
            return counter_;
        }

    private:
        int head_ = 0;
        int tail_ = 0;
        int counter_ = 0;
        OperationLog q_[CAPACITY];
        bool isEmpty_{true};
    };

private:
    bool terminationFlag_ = false;
    bool rebuildFlag_ = false;
    pthread_t rebuildThread_{};
    pthread_mutex_t terminationFlagMutexLock_;
    pthread_mutex_t rebuildPtrMutexLock_;
    pthread_mutex_t workingFlagMutex_;
    pthread_mutex_t searchFlagMutex_;
    pthread_mutex_t rebuildLoggerMutexLock_;
    pthread_mutex_t pointsDeletedRebuildMutexLock_;
    ManualQueue rebuildLogger_;
    PointVector rebuildPclStorage_;
    KdTreeNode** rebuildPtr_ = nullptr;
    int searchMutexCounter_ = 0;

    static void* multiThreadPtr_(void* arg);
    void multiThreadRebuild_();
    void startThread_();
    void stopThread_();
    void runOperation_(KdTreeNode** root, OperationLog operation);

    int treeSizeTmp_ = 0;
    int validNumTmp_ = 0;
    float alphaBalTmp_ = 0.5f;
    float alphaDelTmp_ = 0.0f;
    float deleteCriterionParam_ = 0.5f;
    float balanceCriterionParam_ = 0.7f;
    float downsampleSize_ = 0.2f;
    bool deleteStorageDisabled_ = false;
    KdTreeNode* staticRootNode_ = nullptr;
    PointVector pointsDeleted_;
    PointVector downsampleStorage_;
    PointVector multithreadPointsDeleted_;

    void initTreeNode_(KdTreeNode* root);
    void buildTree_(KdTreeNode** root, int l, int r, PointVector& storage);
    void rebuild_(KdTreeNode** root);
    int deleteByRange_(KdTreeNode** root, BoxPointType boxpoint, bool allowRebuild, bool isDownsample);
    void deleteByPoint_(KdTreeNode** root, PointType point, bool allowRebuild);
    void addByPoint_(KdTreeNode** root, PointType point, bool allowRebuild, int fatherAxis);
    void addByRange_(KdTreeNode** root, BoxPointType boxpoint, bool allowRebuild);
    void search_(KdTreeNode* root, int kNearest, PointType point, ManualHeap& q, float maxDist);
    void searchByRange_(KdTreeNode* root, BoxPointType boxpoint, PointVector& storage);
    void searchByRadius_(KdTreeNode* root, PointType point, float radius, PointVector& storage);
    bool criterionCheck_(KdTreeNode* root);
    void pushDown_(KdTreeNode* root);
    void update_(KdTreeNode* root);
    void deleteTreeNodes_(KdTreeNode** root);
    bool samePoint_(PointType a, PointType b);
    float calcDist_(PointType a, PointType b);
    float calcBoxDist_(KdTreeNode* node, PointType point);
    static bool pointCmpX_(PointType a, PointType b);
    static bool pointCmpY_(PointType a, PointType b);
    static bool pointCmpZ_(PointType a, PointType b);

public:
    explicit KdTree(float deleteParam = 0.5f, float balanceParam = 0.6f, float boxLength = 0.2f);
    ~KdTree();
    KdTree(const KdTree&) = delete;
    KdTree& operator=(const KdTree&) = delete;

    void setDeleteCriterionParam(float deleteParam) {
        deleteCriterionParam_ = deleteParam;
    }

    void setBalanceCriterionParam(float balanceParam) {
        balanceCriterionParam_ = balanceParam;
    }

    void setDownsampleParam(float downsampleParam) {
        downsampleSize_ = downsampleParam;
    }

    void initialize(float deleteParam = 0.5f, float balanceParam = 0.7f, float boxLength = 0.2f);
    int size();
    int validNum();
    void rootAlpha(float& alphaBal, float& alphaDel);
    void build(PointVector pointCloud);
    void nearestSearch(
        PointType point,
        int kNearest,
        PointVector& nearestPoints,
        std::vector<float>& pointDistance,
        float maxDist = INFINITY
    );
    void boxSearch(const BoxPointType& boxOfPoint, PointVector& storage);
    void radiusSearch(PointType point, float radius, PointVector& storage);
    int addPoints(PointVector& pointToAdd, bool downsampleOn);
    void addPointBoxes(std::vector<BoxPointType>& boxPoints);
    void deletePoints(PointVector& pointToDel);
    int deletePointBoxes(std::vector<BoxPointType>& boxPoints);
    void flatten(KdTreeNode* root, PointVector& storage, DeleteStorage storageType);
    void acquireRemovedPoints(PointVector& removedPoints);
    BoxPointType treeRange();
    PointVector pclStorage_;
    KdTreeNode* root_ = nullptr;
    int maxQueueSize_ = 0;
};


template <typename PointType>
KdTree<PointType>::KdTree(float deleteParam, float balanceParam, float boxLength)
{
    deleteCriterionParam_ = deleteParam;
    balanceCriterionParam_ = balanceParam;
    downsampleSize_ = boxLength;
    rebuildLogger_.clear();
    terminationFlag_ = false;
    startThread_();
}

template <typename PointType>
KdTree<PointType>::~KdTree()
{
    stopThread_();
    deleteStorageDisabled_ = true;
    deleteTreeNodes_(&root_);
    PointVector().swap(pclStorage_);
    rebuildLogger_.clear();
}



template <typename PointType>
void KdTree<PointType>::initialize(float deleteParam, float balanceParam, float boxLength)
{
    setDeleteCriterionParam(deleteParam);
    setBalanceCriterionParam(balanceParam);
    setDownsampleParam(boxLength);
}

template <typename PointType>
void KdTree<PointType>::initTreeNode_(KdTreeNode *root)
{
    root->point.x = 0.0f;
    root->point.y = 0.0f;
    root->point.z = 0.0f;
    root->node_range_x[0] = 0.0f;
    root->node_range_x[1] = 0.0f;
    root->node_range_y[0] = 0.0f;
    root->node_range_y[1] = 0.0f;
    root->node_range_z[0] = 0.0f;
    root->node_range_z[1] = 0.0f;
    root->radius_sq = 0.0f;
    root->division_axis = 0;
    root->father_ptr = nullptr;
    root->left_son_ptr = nullptr;
    root->right_son_ptr = nullptr;
    root->TreeSize = 0;
    root->invalid_point_num = 0;
    root->down_del_num = 0;
    root->point_deleted = false;
    root->tree_deleted = false;
    root->need_push_down_to_left = false;
    root->need_push_down_to_right = false;
    root->point_downsample_deleted = false;
    root->working_flag = false;
    pthread_mutex_init(&(root->push_down_mutex_lock), nullptr);
}

template <typename PointType>
int KdTree<PointType>::size()
{
    int s = 0;
    if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_)
    {
        if (root_ != nullptr)
        {
            return root_->TreeSize;
        }
        else
        {
            return 0;
        }
    }
    else
    {
        if (!pthread_mutex_trylock(&workingFlagMutex_))
        {
            s = root_->TreeSize;
            pthread_mutex_unlock(&workingFlagMutex_);
            return s;
        }
        else
        {
            return treeSizeTmp_;
        }
    }
}

template <typename PointType>
BoxPointType KdTree<PointType>::treeRange()
{
    BoxPointType range;
    if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_)
    {
        if (root_ != nullptr)
        {
            range.vertex_min[0] = root_->node_range_x[0];
            range.vertex_min[1] = root_->node_range_y[0];
            range.vertex_min[2] = root_->node_range_z[0];
            range.vertex_max[0] = root_->node_range_x[1];
            range.vertex_max[1] = root_->node_range_y[1];
            range.vertex_max[2] = root_->node_range_z[1];
        }
        else
        {
            std::memset(&range, 0, sizeof(range));
        }
    }
    else
    {
        if (!pthread_mutex_trylock(&workingFlagMutex_))
        {
            range.vertex_min[0] = root_->node_range_x[0];
            range.vertex_min[1] = root_->node_range_y[0];
            range.vertex_min[2] = root_->node_range_z[0];
            range.vertex_max[0] = root_->node_range_x[1];
            range.vertex_max[1] = root_->node_range_y[1];
            range.vertex_max[2] = root_->node_range_z[1];
            pthread_mutex_unlock(&workingFlagMutex_);
        }
        else
        {
            std::memset(&range, 0, sizeof(range));
        }
    }
    return range;
}

template <typename PointType>
int KdTree<PointType>::validNum()
{
    int s = 0;
    if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_)
    {
        if (root_ != nullptr)
            return (root_->TreeSize - root_->invalid_point_num);
        else
            return 0;
    }
    else
    {
        if (!pthread_mutex_trylock(&workingFlagMutex_))
        {
            s = root_->TreeSize - root_->invalid_point_num;
            pthread_mutex_unlock(&workingFlagMutex_);
            return s;
        }
        else
        {
            return -1;
        }
    }
}

template <typename PointType>
void KdTree<PointType>::rootAlpha(float &alphaBal, float &alphaDel)
{
    if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_)
    {
        alphaBal = root_->alpha_bal;
        alphaDel = root_->alpha_del;
        return;
    }
    else
    {
        if (!pthread_mutex_trylock(&workingFlagMutex_))
        {
            alphaBal = root_->alpha_bal;
            alphaDel = root_->alpha_del;
            pthread_mutex_unlock(&workingFlagMutex_);
            return;
        }
        else
        {
            alphaBal = alphaBalTmp_;
            alphaDel = alphaDelTmp_;
            return;
        }
    }
}

template <typename PointType>
void KdTree<PointType>::startThread_()
{
    pthread_mutex_init(&terminationFlagMutexLock_, nullptr);
    pthread_mutex_init(&rebuildPtrMutexLock_, nullptr);
    pthread_mutex_init(&rebuildLoggerMutexLock_, nullptr);
    pthread_mutex_init(&pointsDeletedRebuildMutexLock_, nullptr);
    pthread_mutex_init(&workingFlagMutex_, nullptr);
    pthread_mutex_init(&searchFlagMutex_, nullptr);
    pthread_create(&rebuildThread_, nullptr, multiThreadPtr_, (void *)this);
}

template <typename PointType>
void KdTree<PointType>::stopThread_()
{
    pthread_mutex_lock(&terminationFlagMutexLock_);
    terminationFlag_ = true;
    pthread_mutex_unlock(&terminationFlagMutexLock_);
    if (rebuildThread_)
        pthread_join(rebuildThread_, nullptr);
    pthread_mutex_destroy(&terminationFlagMutexLock_);
    pthread_mutex_destroy(&rebuildLoggerMutexLock_);
    pthread_mutex_destroy(&rebuildPtrMutexLock_);
    pthread_mutex_destroy(&pointsDeletedRebuildMutexLock_);
    pthread_mutex_destroy(&workingFlagMutex_);
    pthread_mutex_destroy(&searchFlagMutex_);
}

template <typename PointType>
void* KdTree<PointType>::multiThreadPtr_(void* arg)
{
    KdTree *handle = (KdTree *)arg;
    handle->multiThreadRebuild_();
    return nullptr;
}

template <typename PointType>
void KdTree<PointType>::multiThreadRebuild_()
{
    bool terminated = false;
    KdTreeNode* father_ptr;
    pthread_mutex_lock(&terminationFlagMutexLock_);
    terminated = terminationFlag_;
    pthread_mutex_unlock(&terminationFlagMutexLock_);
    while (!terminated)
    {
        pthread_mutex_lock(&rebuildPtrMutexLock_);
        pthread_mutex_lock(&workingFlagMutex_);
        if (rebuildPtr_ != nullptr)
        {
            /* Traverse and copy */
            if (!rebuildLogger_.empty())
            {
                std::cerr << "ikd-tree: rebuild logger is not empty\n";
            }
            rebuildFlag_ = true;
            if (*rebuildPtr_ == root_)
            {
                treeSizeTmp_ = root_->TreeSize;
                validNumTmp_ = root_->TreeSize - root_->invalid_point_num;
                alphaBalTmp_ = root_->alpha_bal;
                alphaDelTmp_ = root_->alpha_del;
            }
            KdTreeNode *old_root_node = (*rebuildPtr_);
            father_ptr = (*rebuildPtr_)->father_ptr;
            PointVector().swap(rebuildPclStorage_);
            // Lock search
            pthread_mutex_lock(&searchFlagMutex_);
            while (searchMutexCounter_ != 0)
            {
                pthread_mutex_unlock(&searchFlagMutex_);
                usleep(1);
                pthread_mutex_lock(&searchFlagMutex_);
            }
            searchMutexCounter_ = -1;
            pthread_mutex_unlock(&searchFlagMutex_);
            // Lock deleted points cache
            pthread_mutex_lock(&pointsDeletedRebuildMutexLock_);
            flatten(*rebuildPtr_, rebuildPclStorage_, DeleteStorage::MULTI_THREAD_REC);
            // Unlock deleted points cache
            pthread_mutex_unlock(&pointsDeletedRebuildMutexLock_);
            // Unlock search
            pthread_mutex_lock(&searchFlagMutex_);
            searchMutexCounter_ = 0;
            pthread_mutex_unlock(&searchFlagMutex_);
            pthread_mutex_unlock(&workingFlagMutex_);
            /* Rebuild and update missed operations */
            OperationLog operation;
            KdTreeNode *new_root_node = nullptr;
            if (static_cast<int>(rebuildPclStorage_.size()) > 0)
            {
                buildTree_(&new_root_node, 0, static_cast<int>(rebuildPclStorage_.size()) - 1, rebuildPclStorage_);
                // Rebuild has been done. Updates the blocked operations into the new tree
                pthread_mutex_lock(&workingFlagMutex_);
                pthread_mutex_lock(&rebuildLoggerMutexLock_);
                int tmp_counter = 0;
                while (!rebuildLogger_.empty())
                {
                    operation = rebuildLogger_.front();
                    maxQueueSize_ = std::max(maxQueueSize_, rebuildLogger_.size());
                    rebuildLogger_.pop();
                    pthread_mutex_unlock(&rebuildLoggerMutexLock_);
                    pthread_mutex_unlock(&workingFlagMutex_);
                    runOperation_(&new_root_node, operation);
                    tmp_counter++;
                    if (tmp_counter % 10 == 0)
                        usleep(1);
                    pthread_mutex_lock(&workingFlagMutex_);
                    pthread_mutex_lock(&rebuildLoggerMutexLock_);
                }
                pthread_mutex_unlock(&rebuildLoggerMutexLock_);
            }
            /* Replace to original tree*/
            // pthread_mutex_lock(&workingFlagMutex_);
            pthread_mutex_lock(&searchFlagMutex_);
            while (searchMutexCounter_ != 0)
            {
                pthread_mutex_unlock(&searchFlagMutex_);
                usleep(1);
                pthread_mutex_lock(&searchFlagMutex_);
            }
            searchMutexCounter_ = -1;
            pthread_mutex_unlock(&searchFlagMutex_);
            if (father_ptr->left_son_ptr == *rebuildPtr_)
            {
                father_ptr->left_son_ptr = new_root_node;
            }
            else if (father_ptr->right_son_ptr == *rebuildPtr_)
            {
                father_ptr->right_son_ptr = new_root_node;
            }
            else
            {
                throw std::runtime_error("Father ptr incompatible with current node");
            }
            if (new_root_node != nullptr)
                new_root_node->father_ptr = father_ptr;
            (*rebuildPtr_) = new_root_node;
            if (father_ptr == staticRootNode_)
                root_ = staticRootNode_->left_son_ptr;
            KdTreeNode *update_root = *rebuildPtr_;
            while (update_root != nullptr && update_root != root_)
            {
                update_root = update_root->father_ptr;
                if (update_root->working_flag)
                    break;
                if (update_root == update_root->father_ptr->left_son_ptr && update_root->father_ptr->need_push_down_to_left)
                    break;
                if (update_root == update_root->father_ptr->right_son_ptr && update_root->father_ptr->need_push_down_to_right)
                    break;
                update_(update_root);
            }
            pthread_mutex_lock(&searchFlagMutex_);
            searchMutexCounter_ = 0;
            pthread_mutex_unlock(&searchFlagMutex_);
            rebuildPtr_ = nullptr;
            pthread_mutex_unlock(&workingFlagMutex_);
            rebuildFlag_ = false;
            /* Delete discarded tree nodes */
            deleteTreeNodes_(&old_root_node);
        }
        else
        {
            pthread_mutex_unlock(&workingFlagMutex_);
        }
        pthread_mutex_unlock(&rebuildPtrMutexLock_);
        pthread_mutex_lock(&terminationFlagMutexLock_);
        terminated = terminationFlag_;
        pthread_mutex_unlock(&terminationFlagMutexLock_);
        usleep(100);
    }
}

template <typename PointType>
void KdTree<PointType>::runOperation_(KdTreeNode **root, OperationLog operation)
{
    switch (operation.op)
    {
    case Operation::ADD_POINT:
        addByPoint_(root, operation.point, false, (*root)->division_axis);
        break;
    case Operation::ADD_BOX:
        addByRange_(root, operation.boxpoint, false);
        break;
    case Operation::DELETE_POINT:
        deleteByPoint_(root, operation.point, false);
        break;
    case Operation::DELETE_BOX:
        deleteByRange_(root, operation.boxpoint, false, false);
        break;
    case Operation::DOWNSAMPLE_DELETE:
        deleteByRange_(root, operation.boxpoint, false, true);
        break;
    case Operation::PUSH_DOWN:
        (*root)->tree_downsample_deleted |= operation.tree_downsample_deleted;
        (*root)->point_downsample_deleted |= operation.tree_downsample_deleted;
        (*root)->tree_deleted = operation.tree_deleted || (*root)->tree_downsample_deleted;
        (*root)->point_deleted = (*root)->tree_deleted || (*root)->point_downsample_deleted;
        if (operation.tree_downsample_deleted)
            (*root)->down_del_num = (*root)->TreeSize;
        if (operation.tree_deleted)
            (*root)->invalid_point_num = (*root)->TreeSize;
        else
            (*root)->invalid_point_num = (*root)->down_del_num;
        (*root)->need_push_down_to_left = true;
        (*root)->need_push_down_to_right = true;
        break;
    default:
        break;
    }
}

template <typename PointType>
void KdTree<PointType>::build(PointVector pointCloud)
{
    if (root_ != nullptr)
    {
        deleteTreeNodes_(&root_);
    }
    if (pointCloud.size() == 0)
        return;
    staticRootNode_ = new KdTreeNode;
    initTreeNode_(staticRootNode_);
    buildTree_(&staticRootNode_->left_son_ptr, 0, pointCloud.size() - 1, pointCloud);
    update_(staticRootNode_);
    staticRootNode_->TreeSize = 0;
    root_ = staticRootNode_->left_son_ptr;
}

template <typename PointType>
void KdTree<PointType>::nearestSearch(PointType point, int kNearest, PointVector &nearestPoints, std::vector<float> &pointDistance, float maxDist)
{
    ManualHeap q(2 * kNearest);
    q.clear();
    std::vector<float>().swap(pointDistance);
    if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_)
    {
        search_(root_, kNearest, point, q, maxDist);
    }
    else
    {
        pthread_mutex_lock(&searchFlagMutex_);
        while (searchMutexCounter_ == -1)
        {
            pthread_mutex_unlock(&searchFlagMutex_);
            usleep(1);
            pthread_mutex_lock(&searchFlagMutex_);
        }
        searchMutexCounter_ += 1;
        pthread_mutex_unlock(&searchFlagMutex_);
        search_(root_, kNearest, point, q, maxDist);
        pthread_mutex_lock(&searchFlagMutex_);
        searchMutexCounter_ -= 1;
        pthread_mutex_unlock(&searchFlagMutex_);
    }
    int k_found = std::min(kNearest, static_cast<int>(q.size()));
    PointVector().swap(nearestPoints);
    std::vector<float>().swap(pointDistance);
    for (int i = 0; i < k_found; i++)
    {
        nearestPoints.insert(nearestPoints.begin(), q.top().point);
        pointDistance.insert(pointDistance.begin(), q.top().dist);
        q.pop();
    }
    return;
}

template <typename PointType>
void KdTree<PointType>::boxSearch(const BoxPointType &boxOfPoint, PointVector &storage)
{
    storage.clear();
    searchByRange_(root_, boxOfPoint, storage);
}

template <typename PointType>
void KdTree<PointType>::radiusSearch(PointType point, float radius, PointVector &storage)
{
    storage.clear();
    searchByRadius_(root_, point, radius, storage);
}

template <typename PointType>
int KdTree<PointType>::addPoints(PointVector &pointToAdd, bool downsampleOn)
{
    int NewPointSize = pointToAdd.size();
    int tree_size = size();
    BoxPointType boxOfPoint;
    PointType downsample_result, mid_point;
    bool downsample_switch = downsampleOn && DOWNSAMPLE_SWITCH;
    float min_dist, tmp_dist;
    int tmp_counter = 0;
    for (int i = 0; i < pointToAdd.size(); i++)
    {
        if (downsample_switch)
        {
            boxOfPoint.vertex_min[0] = std::floor(pointToAdd[i].x / downsampleSize_) * downsampleSize_;
            boxOfPoint.vertex_max[0] = boxOfPoint.vertex_min[0] + downsampleSize_;
            boxOfPoint.vertex_min[1] = std::floor(pointToAdd[i].y / downsampleSize_) * downsampleSize_;
            boxOfPoint.vertex_max[1] = boxOfPoint.vertex_min[1] + downsampleSize_;
            boxOfPoint.vertex_min[2] = std::floor(pointToAdd[i].z / downsampleSize_) * downsampleSize_;
            boxOfPoint.vertex_max[2] = boxOfPoint.vertex_min[2] + downsampleSize_;
            mid_point.x = boxOfPoint.vertex_min[0] + (boxOfPoint.vertex_max[0] - boxOfPoint.vertex_min[0]) / 2.0;
            mid_point.y = boxOfPoint.vertex_min[1] + (boxOfPoint.vertex_max[1] - boxOfPoint.vertex_min[1]) / 2.0;
            mid_point.z = boxOfPoint.vertex_min[2] + (boxOfPoint.vertex_max[2] - boxOfPoint.vertex_min[2]) / 2.0;
            PointVector().swap(downsampleStorage_);
            searchByRange_(root_, boxOfPoint, downsampleStorage_);
            min_dist = calcDist_(pointToAdd[i], mid_point);
            downsample_result = pointToAdd[i];
            for (int index = 0; index < downsampleStorage_.size(); index++)
            {
                tmp_dist = calcDist_(downsampleStorage_[index], mid_point);
                if (tmp_dist < min_dist)
                {
                    min_dist = tmp_dist;
                    downsample_result = downsampleStorage_[index];
                }
            }
            if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_)
            {
                if (downsampleStorage_.size() > 1 || samePoint_(pointToAdd[i], downsample_result))
                {
                    if (downsampleStorage_.size() > 0)
                        deleteByRange_(&root_, boxOfPoint, true, true);
                    addByPoint_(&root_, downsample_result, true, root_->division_axis);
                    tmp_counter++;
                }
            }
            else
            {
                if (downsampleStorage_.size() > 1 || samePoint_(pointToAdd[i], downsample_result))
                {
                    OperationLog operationDelete, operation;
                    operationDelete.boxpoint = boxOfPoint;
                    operationDelete.op = Operation::DOWNSAMPLE_DELETE;
                    operation.point = downsample_result;
                    operation.op = Operation::ADD_POINT;
                    pthread_mutex_lock(&workingFlagMutex_);
                    if (downsampleStorage_.size() > 0)
                        deleteByRange_(&root_, boxOfPoint, false, true);
                    addByPoint_(&root_, downsample_result, false, root_->division_axis);
                    tmp_counter++;
                    if (rebuildFlag_)
                    {
                        pthread_mutex_lock(&rebuildLoggerMutexLock_);
                        if (downsampleStorage_.size() > 0)
                            rebuildLogger_.push(operationDelete);
                        rebuildLogger_.push(operation);
                        pthread_mutex_unlock(&rebuildLoggerMutexLock_);
                    }
                    pthread_mutex_unlock(&workingFlagMutex_);
                };
            }
        }
        else
        {
            if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_)
            {
                addByPoint_(&root_, pointToAdd[i], true, root_->division_axis);
            }
            else
            {
                OperationLog operation;
                operation.point = pointToAdd[i];
                operation.op = Operation::ADD_POINT;
                pthread_mutex_lock(&workingFlagMutex_);
                addByPoint_(&root_, pointToAdd[i], false, root_->division_axis);
                if (rebuildFlag_)
                {
                    pthread_mutex_lock(&rebuildLoggerMutexLock_);
                    rebuildLogger_.push(operation);
                    pthread_mutex_unlock(&rebuildLoggerMutexLock_);
                }
                pthread_mutex_unlock(&workingFlagMutex_);
            }
        }
    }
    return tmp_counter;
}

template <typename PointType>
void KdTree<PointType>::addPointBoxes(std::vector<BoxPointType> &boxPoints)
{
    for (int i = 0; i < boxPoints.size(); i++)
    {
        if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_)
        {
            addByRange_(&root_, boxPoints[i], true);
        }
        else
        {
            OperationLog operation;
            operation.boxpoint = boxPoints[i];
            operation.op = Operation::ADD_BOX;
            pthread_mutex_lock(&workingFlagMutex_);
            addByRange_(&root_, boxPoints[i], false);
            if (rebuildFlag_)
            {
                pthread_mutex_lock(&rebuildLoggerMutexLock_);
                rebuildLogger_.push(operation);
                pthread_mutex_unlock(&rebuildLoggerMutexLock_);
            }
            pthread_mutex_unlock(&workingFlagMutex_);
        }
    }
    return;
}

template <typename PointType>
void KdTree<PointType>::deletePoints(PointVector &pointToDel)
{
    for (int i = 0; i < pointToDel.size(); i++)
    {
        if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_)
        {
            deleteByPoint_(&root_, pointToDel[i], true);
        }
        else
        {
            OperationLog operation;
            operation.point = pointToDel[i];
            operation.op = Operation::DELETE_POINT;
            pthread_mutex_lock(&workingFlagMutex_);
            deleteByPoint_(&root_, pointToDel[i], false);
            if (rebuildFlag_)
            {
                pthread_mutex_lock(&rebuildLoggerMutexLock_);
                rebuildLogger_.push(operation);
                pthread_mutex_unlock(&rebuildLoggerMutexLock_);
            }
            pthread_mutex_unlock(&workingFlagMutex_);
        }
    }
    return;
}

template <typename PointType>
int KdTree<PointType>::deletePointBoxes(std::vector<BoxPointType> &boxPoints)
{
    int tmp_counter = 0;
    for (int i = 0; i < boxPoints.size(); i++)
    {
        if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_)
        {
            tmp_counter += deleteByRange_(&root_, boxPoints[i], true, false);
        }
        else
        {
            OperationLog operation;
            operation.boxpoint = boxPoints[i];
            operation.op = Operation::DELETE_BOX;
            pthread_mutex_lock(&workingFlagMutex_);
            tmp_counter += deleteByRange_(&root_, boxPoints[i], false, false);
            if (rebuildFlag_)
            {
                pthread_mutex_lock(&rebuildLoggerMutexLock_);
                rebuildLogger_.push(operation);
                pthread_mutex_unlock(&rebuildLoggerMutexLock_);
            }
            pthread_mutex_unlock(&workingFlagMutex_);
        }
    }
    return tmp_counter;
}

template <typename PointType>
void KdTree<PointType>::acquireRemovedPoints(PointVector &removedPoints)
{
    pthread_mutex_lock(&pointsDeletedRebuildMutexLock_);
    for (int i = 0; i < pointsDeleted_.size(); i++)
    {
        removedPoints.push_back(pointsDeleted_[i]);
    }
    for (int i = 0; i < multithreadPointsDeleted_.size(); i++)
    {
        removedPoints.push_back(multithreadPointsDeleted_[i]);
    }
    pointsDeleted_.clear();
    multithreadPointsDeleted_.clear();
    pthread_mutex_unlock(&pointsDeletedRebuildMutexLock_);
    return;
}

template <typename PointType>
void KdTree<PointType>::buildTree_(KdTreeNode **root, int l, int r, PointVector &storage)
{
    if (l > r)
        return;
    *root = new KdTreeNode;
    initTreeNode_(*root);
    int mid = (l + r) >> 1;
    int div_axis = 0;
    int i;
    // Find the best division Axis
    float min_value[3] = {INFINITY, INFINITY, INFINITY};
    float max_value[3] = {-INFINITY, -INFINITY, -INFINITY};
    float dim_range[3] = {0, 0, 0};
    for (i = l; i <= r; i++)
    {
        min_value[0] = std::min(min_value[0], storage[i].x);
        min_value[1] = std::min(min_value[1], storage[i].y);
        min_value[2] = std::min(min_value[2], storage[i].z);
        max_value[0] = std::max(max_value[0], storage[i].x);
        max_value[1] = std::max(max_value[1], storage[i].y);
        max_value[2] = std::max(max_value[2], storage[i].z);
    }
    // Select the longest dimension as division axis
    for (i = 0; i < 3; i++)
        dim_range[i] = max_value[i] - min_value[i];
    for (i = 1; i < 3; i++)
        if (dim_range[i] > dim_range[div_axis])
            div_axis = i;
    // Divide by the division axis and recursively build.

    (*root)->division_axis = div_axis;
    switch (div_axis)
    {
    case 0:
        std::nth_element(std::begin(storage) + l, std::begin(storage) + mid, std::begin(storage) + r + 1, pointCmpX_);
        break;
    case 1:
        std::nth_element(std::begin(storage) + l, std::begin(storage) + mid, std::begin(storage) + r + 1, pointCmpY_);
        break;
    case 2:
        std::nth_element(std::begin(storage) + l, std::begin(storage) + mid, std::begin(storage) + r + 1, pointCmpZ_);
        break;
    default:
        std::nth_element(std::begin(storage) + l, std::begin(storage) + mid, std::begin(storage) + r + 1, pointCmpX_);
        break;
    }
    (*root)->point = storage[mid];
    KdTreeNode *left_son = nullptr, *right_son = nullptr;
    buildTree_(&left_son, l, mid - 1, storage);
    buildTree_(&right_son, mid + 1, r, storage);
    (*root)->left_son_ptr = left_son;
    (*root)->right_son_ptr = right_son;
    update_((*root));
    return;
}

template <typename PointType>
void KdTree<PointType>::rebuild_(KdTreeNode **root)
{
    KdTreeNode *father_ptr;
    if ((*root)->TreeSize >= MULTI_THREAD_REBUILD_POINT_NUM)
    {
        if (!pthread_mutex_trylock(&rebuildPtrMutexLock_))
        {
            if (rebuildPtr_ == nullptr || ((*root)->TreeSize > (*rebuildPtr_)->TreeSize))
            {
                rebuildPtr_ = root;
            }
            pthread_mutex_unlock(&rebuildPtrMutexLock_);
        }
    }
    else
    {
        father_ptr = (*root)->father_ptr;
        pclStorage_.clear();
        flatten(*root, pclStorage_, DeleteStorage::DELETE_POINTS_REC);
        deleteTreeNodes_(root);
        buildTree_(root, 0, pclStorage_.size() - 1, pclStorage_);
        if (*root != nullptr)
            (*root)->father_ptr = father_ptr;
        if (*root == root_)
            staticRootNode_->left_son_ptr = *root;
    }
    return;
}

template <typename PointType>
int KdTree<PointType>::deleteByRange_(KdTreeNode **root, BoxPointType boxpoint, bool allowRebuild, bool isDownsample)
{
    if ((*root) == nullptr || (*root)->tree_deleted)
        return 0;
    (*root)->working_flag = true;
    pushDown_(*root);
    int tmp_counter = 0;
    if (boxpoint.vertex_max[0] <= (*root)->node_range_x[0] || boxpoint.vertex_min[0] > (*root)->node_range_x[1])
        return 0;
    if (boxpoint.vertex_max[1] <= (*root)->node_range_y[0] || boxpoint.vertex_min[1] > (*root)->node_range_y[1])
        return 0;
    if (boxpoint.vertex_max[2] <= (*root)->node_range_z[0] || boxpoint.vertex_min[2] > (*root)->node_range_z[1])
        return 0;
    if (boxpoint.vertex_min[0] <= (*root)->node_range_x[0] && boxpoint.vertex_max[0] > (*root)->node_range_x[1] && boxpoint.vertex_min[1] <= (*root)->node_range_y[0] && boxpoint.vertex_max[1] > (*root)->node_range_y[1] && boxpoint.vertex_min[2] <= (*root)->node_range_z[0] && boxpoint.vertex_max[2] > (*root)->node_range_z[1])
    {
        (*root)->tree_deleted = true;
        (*root)->point_deleted = true;
        (*root)->need_push_down_to_left = true;
        (*root)->need_push_down_to_right = true;
        tmp_counter = (*root)->TreeSize - (*root)->invalid_point_num;
        (*root)->invalid_point_num = (*root)->TreeSize;
        if (isDownsample)
        {
            (*root)->tree_downsample_deleted = true;
            (*root)->point_downsample_deleted = true;
            (*root)->down_del_num = (*root)->TreeSize;
        }
        return tmp_counter;
    }
    if (!(*root)->point_deleted && boxpoint.vertex_min[0] <= (*root)->point.x && boxpoint.vertex_max[0] > (*root)->point.x && boxpoint.vertex_min[1] <= (*root)->point.y && boxpoint.vertex_max[1] > (*root)->point.y && boxpoint.vertex_min[2] <= (*root)->point.z && boxpoint.vertex_max[2] > (*root)->point.z)
    {
        (*root)->point_deleted = true;
        tmp_counter += 1;
        if (isDownsample)
            (*root)->point_downsample_deleted = true;
    }
    OperationLog delete_box_log;
    if (isDownsample)
        delete_box_log.op = Operation::DOWNSAMPLE_DELETE;
    else
        delete_box_log.op = Operation::DELETE_BOX;
    delete_box_log.boxpoint = boxpoint;
    if ((rebuildPtr_ == nullptr) || (*root)->left_son_ptr != *rebuildPtr_)
    {
        tmp_counter += deleteByRange_(&((*root)->left_son_ptr), boxpoint, allowRebuild, isDownsample);
    }
    else
    {
        pthread_mutex_lock(&workingFlagMutex_);
        tmp_counter += deleteByRange_(&((*root)->left_son_ptr), boxpoint, false, isDownsample);
        if (rebuildFlag_)
        {
            pthread_mutex_lock(&rebuildLoggerMutexLock_);
            rebuildLogger_.push(delete_box_log);
            pthread_mutex_unlock(&rebuildLoggerMutexLock_);
        }
        pthread_mutex_unlock(&workingFlagMutex_);
    }
    if ((rebuildPtr_ == nullptr) || (*root)->right_son_ptr != *rebuildPtr_)
    {
        tmp_counter += deleteByRange_(&((*root)->right_son_ptr), boxpoint, allowRebuild, isDownsample);
    }
    else
    {
        pthread_mutex_lock(&workingFlagMutex_);
        tmp_counter += deleteByRange_(&((*root)->right_son_ptr), boxpoint, false, isDownsample);
        if (rebuildFlag_)
        {
            pthread_mutex_lock(&rebuildLoggerMutexLock_);
            rebuildLogger_.push(delete_box_log);
            pthread_mutex_unlock(&rebuildLoggerMutexLock_);
        }
        pthread_mutex_unlock(&workingFlagMutex_);
    }
    update_(*root);
    if (rebuildPtr_ != nullptr && *rebuildPtr_ == *root && (*root)->TreeSize < MULTI_THREAD_REBUILD_POINT_NUM)
        rebuildPtr_ = nullptr;
    bool need_rebuild = allowRebuild & criterionCheck_((*root));
    if (need_rebuild)
        rebuild_(root);
    if ((*root) != nullptr)
        (*root)->working_flag = false;
    return tmp_counter;
}

template <typename PointType>
void KdTree<PointType>::deleteByPoint_(KdTreeNode **root, PointType point, bool allowRebuild)
{
    if ((*root) == nullptr || (*root)->tree_deleted)
        return;
    (*root)->working_flag = true;
    pushDown_(*root);
    if (samePoint_((*root)->point, point) && !(*root)->point_deleted)
    {
        (*root)->point_deleted = true;
        (*root)->invalid_point_num += 1;
        if ((*root)->invalid_point_num == (*root)->TreeSize)
            (*root)->tree_deleted = true;
        return;
    }
    OperationLog delete_log;
    delete_log.op = Operation::DELETE_POINT;
    delete_log.point = point;
    if (((*root)->division_axis == 0 && point.x < (*root)->point.x) || ((*root)->division_axis == 1 && point.y < (*root)->point.y) || ((*root)->division_axis == 2 && point.z < (*root)->point.z))
    {
        if ((rebuildPtr_ == nullptr) || (*root)->left_son_ptr != *rebuildPtr_)
        {
            deleteByPoint_(&(*root)->left_son_ptr, point, allowRebuild);
        }
        else
        {
            pthread_mutex_lock(&workingFlagMutex_);
            deleteByPoint_(&(*root)->left_son_ptr, point, false);
            if (rebuildFlag_)
            {
                pthread_mutex_lock(&rebuildLoggerMutexLock_);
                rebuildLogger_.push(delete_log);
                pthread_mutex_unlock(&rebuildLoggerMutexLock_);
            }
            pthread_mutex_unlock(&workingFlagMutex_);
        }
    }
    else
    {
        if ((rebuildPtr_ == nullptr) || (*root)->right_son_ptr != *rebuildPtr_)
        {
            deleteByPoint_(&(*root)->right_son_ptr, point, allowRebuild);
        }
        else
        {
            pthread_mutex_lock(&workingFlagMutex_);
            deleteByPoint_(&(*root)->right_son_ptr, point, false);
            if (rebuildFlag_)
            {
                pthread_mutex_lock(&rebuildLoggerMutexLock_);
                rebuildLogger_.push(delete_log);
                pthread_mutex_unlock(&rebuildLoggerMutexLock_);
            }
            pthread_mutex_unlock(&workingFlagMutex_);
        }
    }
    update_(*root);
    if (rebuildPtr_ != nullptr && *rebuildPtr_ == *root && (*root)->TreeSize < MULTI_THREAD_REBUILD_POINT_NUM)
        rebuildPtr_ = nullptr;
    bool need_rebuild = allowRebuild & criterionCheck_((*root));
    if (need_rebuild)
        rebuild_(root);
    if ((*root) != nullptr)
        (*root)->working_flag = false;
    return;
}

template <typename PointType>
void KdTree<PointType>::addByRange_(KdTreeNode **root, BoxPointType boxpoint, bool allowRebuild)
{
    if ((*root) == nullptr)
        return;
    (*root)->working_flag = true;
    pushDown_(*root);
    if (boxpoint.vertex_max[0] <= (*root)->node_range_x[0] || boxpoint.vertex_min[0] > (*root)->node_range_x[1])
        return;
    if (boxpoint.vertex_max[1] <= (*root)->node_range_y[0] || boxpoint.vertex_min[1] > (*root)->node_range_y[1])
        return;
    if (boxpoint.vertex_max[2] <= (*root)->node_range_z[0] || boxpoint.vertex_min[2] > (*root)->node_range_z[1])
        return;
    if (boxpoint.vertex_min[0] <= (*root)->node_range_x[0] && boxpoint.vertex_max[0] > (*root)->node_range_x[1] && boxpoint.vertex_min[1] <= (*root)->node_range_y[0] && boxpoint.vertex_max[1] > (*root)->node_range_y[1] && boxpoint.vertex_min[2] <= (*root)->node_range_z[0] && boxpoint.vertex_max[2] > (*root)->node_range_z[1])
    {
        (*root)->tree_deleted = false || (*root)->tree_downsample_deleted;
        (*root)->point_deleted = false || (*root)->point_downsample_deleted;
        (*root)->need_push_down_to_left = true;
        (*root)->need_push_down_to_right = true;
        (*root)->invalid_point_num = (*root)->down_del_num;
        return;
    }
    if (boxpoint.vertex_min[0] <= (*root)->point.x && boxpoint.vertex_max[0] > (*root)->point.x && boxpoint.vertex_min[1] <= (*root)->point.y && boxpoint.vertex_max[1] > (*root)->point.y && boxpoint.vertex_min[2] <= (*root)->point.z && boxpoint.vertex_max[2] > (*root)->point.z)
    {
        (*root)->point_deleted = (*root)->point_downsample_deleted;
    }
    OperationLog add_box_log;
    add_box_log.op = Operation::ADD_BOX;
    add_box_log.boxpoint = boxpoint;
    if ((rebuildPtr_ == nullptr) || (*root)->left_son_ptr != *rebuildPtr_)
    {
        addByRange_(&((*root)->left_son_ptr), boxpoint, allowRebuild);
    }
    else
    {
        pthread_mutex_lock(&workingFlagMutex_);
        addByRange_(&((*root)->left_son_ptr), boxpoint, false);
        if (rebuildFlag_)
        {
            pthread_mutex_lock(&rebuildLoggerMutexLock_);
            rebuildLogger_.push(add_box_log);
            pthread_mutex_unlock(&rebuildLoggerMutexLock_);
        }
        pthread_mutex_unlock(&workingFlagMutex_);
    }
    if ((rebuildPtr_ == nullptr) || (*root)->right_son_ptr != *rebuildPtr_)
    {
        addByRange_(&((*root)->right_son_ptr), boxpoint, allowRebuild);
    }
    else
    {
        pthread_mutex_lock(&workingFlagMutex_);
        addByRange_(&((*root)->right_son_ptr), boxpoint, false);
        if (rebuildFlag_)
        {
            pthread_mutex_lock(&rebuildLoggerMutexLock_);
            rebuildLogger_.push(add_box_log);
            pthread_mutex_unlock(&rebuildLoggerMutexLock_);
        }
        pthread_mutex_unlock(&workingFlagMutex_);
    }
    update_(*root);
    if (rebuildPtr_ != nullptr && *rebuildPtr_ == *root && (*root)->TreeSize < MULTI_THREAD_REBUILD_POINT_NUM)
        rebuildPtr_ = nullptr;
    bool need_rebuild = allowRebuild & criterionCheck_((*root));
    if (need_rebuild)
        rebuild_(root);
    if ((*root) != nullptr)
        (*root)->working_flag = false;
    return;
}

template <typename PointType>
void KdTree<PointType>::addByPoint_(KdTreeNode **root, PointType point, bool allowRebuild, int fatherAxis)
{
    if (*root == nullptr)
    {
        *root = new KdTreeNode;
        initTreeNode_(*root);
        (*root)->point = point;
        (*root)->division_axis = (fatherAxis + 1) % 3;
        update_(*root);
        return;
    }
    (*root)->working_flag = true;
    OperationLog add_log;
    add_log.op = Operation::ADD_POINT;
    add_log.point = point;
    pushDown_(*root);
    if (((*root)->division_axis == 0 && point.x < (*root)->point.x) || ((*root)->division_axis == 1 && point.y < (*root)->point.y) || ((*root)->division_axis == 2 && point.z < (*root)->point.z))
    {
        if ((rebuildPtr_ == nullptr) || (*root)->left_son_ptr != *rebuildPtr_)
        {
            addByPoint_(&(*root)->left_son_ptr, point, allowRebuild, (*root)->division_axis);
        }
        else
        {
            pthread_mutex_lock(&workingFlagMutex_);
            addByPoint_(&(*root)->left_son_ptr, point, false, (*root)->division_axis);
            if (rebuildFlag_)
            {
                pthread_mutex_lock(&rebuildLoggerMutexLock_);
                rebuildLogger_.push(add_log);
                pthread_mutex_unlock(&rebuildLoggerMutexLock_);
            }
            pthread_mutex_unlock(&workingFlagMutex_);
        }
    }
    else
    {
        if ((rebuildPtr_ == nullptr) || (*root)->right_son_ptr != *rebuildPtr_)
        {
            addByPoint_(&(*root)->right_son_ptr, point, allowRebuild, (*root)->division_axis);
        }
        else
        {
            pthread_mutex_lock(&workingFlagMutex_);
            addByPoint_(&(*root)->right_son_ptr, point, false, (*root)->division_axis);
            if (rebuildFlag_)
            {
                pthread_mutex_lock(&rebuildLoggerMutexLock_);
                rebuildLogger_.push(add_log);
                pthread_mutex_unlock(&rebuildLoggerMutexLock_);
            }
            pthread_mutex_unlock(&workingFlagMutex_);
        }
    }
    update_(*root);
    if (rebuildPtr_ != nullptr && *rebuildPtr_ == *root && (*root)->TreeSize < MULTI_THREAD_REBUILD_POINT_NUM)
        rebuildPtr_ = nullptr;
    bool need_rebuild = allowRebuild & criterionCheck_((*root));
    if (need_rebuild)
        rebuild_(root);
    if ((*root) != nullptr)
        (*root)->working_flag = false;
    return;
}

template <typename PointType>
void KdTree<PointType>::search_(KdTreeNode *root, int kNearest, PointType point, ManualHeap &q, float maxDist)
{
    if (root == nullptr || root->tree_deleted)
        return;
    float cur_dist = calcBoxDist_(root, point);
    float max_dist_sqr = maxDist * maxDist;
    if (cur_dist > max_dist_sqr)
        return;
    int retval;
    if (root->need_push_down_to_left || root->need_push_down_to_right)
    {
        retval = pthread_mutex_trylock(&(root->push_down_mutex_lock));
        if (retval == 0)
        {
            pushDown_(root);
            pthread_mutex_unlock(&(root->push_down_mutex_lock));
        }
        else
        {
            pthread_mutex_lock(&(root->push_down_mutex_lock));
            pthread_mutex_unlock(&(root->push_down_mutex_lock));
        }
    }
    if (!root->point_deleted)
    {
        float dist = calcDist_(point, root->point);
        if (dist <= max_dist_sqr && (q.size() < kNearest || dist < q.top().dist))
        {
            if (q.size() >= kNearest)
                q.pop();
            PointTypeCmp current_point{root->point, dist};
            q.push(current_point);
        }
    }
    int cur_search_counter;
    float dist_left_node = calcBoxDist_(root->left_son_ptr, point);
    float dist_right_node = calcBoxDist_(root->right_son_ptr, point);
    if (q.size() < kNearest || (dist_left_node < q.top().dist && dist_right_node < q.top().dist))
    {
        if (dist_left_node <= dist_right_node)
        {
            if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->left_son_ptr)
            {
                search_(root->left_son_ptr, kNearest, point, q, maxDist);
            }
            else
            {
                pthread_mutex_lock(&searchFlagMutex_);
                while (searchMutexCounter_ == -1)
                {
                    pthread_mutex_unlock(&searchFlagMutex_);
                    usleep(1);
                    pthread_mutex_lock(&searchFlagMutex_);
                }
                searchMutexCounter_ += 1;
                pthread_mutex_unlock(&searchFlagMutex_);
                search_(root->left_son_ptr, kNearest, point, q, maxDist);
                pthread_mutex_lock(&searchFlagMutex_);
                searchMutexCounter_ -= 1;
                pthread_mutex_unlock(&searchFlagMutex_);
            }
            if (q.size() < kNearest || dist_right_node < q.top().dist)
            {
                if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->right_son_ptr)
                {
                    search_(root->right_son_ptr, kNearest, point, q, maxDist);
                }
                else
                {
                    pthread_mutex_lock(&searchFlagMutex_);
                    while (searchMutexCounter_ == -1)
                    {
                        pthread_mutex_unlock(&searchFlagMutex_);
                        usleep(1);
                        pthread_mutex_lock(&searchFlagMutex_);
                    }
                    searchMutexCounter_ += 1;
                    pthread_mutex_unlock(&searchFlagMutex_);
                    search_(root->right_son_ptr, kNearest, point, q, maxDist);
                    pthread_mutex_lock(&searchFlagMutex_);
                    searchMutexCounter_ -= 1;
                    pthread_mutex_unlock(&searchFlagMutex_);
                }
            }
        }
        else
        {
            if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->right_son_ptr)
            {
                search_(root->right_son_ptr, kNearest, point, q, maxDist);
            }
            else
            {
                pthread_mutex_lock(&searchFlagMutex_);
                while (searchMutexCounter_ == -1)
                {
                    pthread_mutex_unlock(&searchFlagMutex_);
                    usleep(1);
                    pthread_mutex_lock(&searchFlagMutex_);
                }
                searchMutexCounter_ += 1;
                pthread_mutex_unlock(&searchFlagMutex_);
                search_(root->right_son_ptr, kNearest, point, q, maxDist);
                pthread_mutex_lock(&searchFlagMutex_);
                searchMutexCounter_ -= 1;
                pthread_mutex_unlock(&searchFlagMutex_);
            }
            if (q.size() < kNearest || dist_left_node < q.top().dist)
            {
                if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->left_son_ptr)
                {
                    search_(root->left_son_ptr, kNearest, point, q, maxDist);
                }
                else
                {
                    pthread_mutex_lock(&searchFlagMutex_);
                    while (searchMutexCounter_ == -1)
                    {
                        pthread_mutex_unlock(&searchFlagMutex_);
                        usleep(1);
                        pthread_mutex_lock(&searchFlagMutex_);
                    }
                    searchMutexCounter_ += 1;
                    pthread_mutex_unlock(&searchFlagMutex_);
                    search_(root->left_son_ptr, kNearest, point, q, maxDist);
                    pthread_mutex_lock(&searchFlagMutex_);
                    searchMutexCounter_ -= 1;
                    pthread_mutex_unlock(&searchFlagMutex_);
                }
            }
        }
    }
    else
    {
        if (dist_left_node < q.top().dist)
        {
            if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->left_son_ptr)
            {
                search_(root->left_son_ptr, kNearest, point, q, maxDist);
            }
            else
            {
                pthread_mutex_lock(&searchFlagMutex_);
                while (searchMutexCounter_ == -1)
                {
                    pthread_mutex_unlock(&searchFlagMutex_);
                    usleep(1);
                    pthread_mutex_lock(&searchFlagMutex_);
                }
                searchMutexCounter_ += 1;
                pthread_mutex_unlock(&searchFlagMutex_);
                search_(root->left_son_ptr, kNearest, point, q, maxDist);
                pthread_mutex_lock(&searchFlagMutex_);
                searchMutexCounter_ -= 1;
                pthread_mutex_unlock(&searchFlagMutex_);
            }
        }
        if (dist_right_node < q.top().dist)
        {
            if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->right_son_ptr)
            {
                search_(root->right_son_ptr, kNearest, point, q, maxDist);
            }
            else
            {
                pthread_mutex_lock(&searchFlagMutex_);
                while (searchMutexCounter_ == -1)
                {
                    pthread_mutex_unlock(&searchFlagMutex_);
                    usleep(1);
                    pthread_mutex_lock(&searchFlagMutex_);
                }
                searchMutexCounter_ += 1;
                pthread_mutex_unlock(&searchFlagMutex_);
                search_(root->right_son_ptr, kNearest, point, q, maxDist);
                pthread_mutex_lock(&searchFlagMutex_);
                searchMutexCounter_ -= 1;
                pthread_mutex_unlock(&searchFlagMutex_);
            }
        }
    }
    return;
}

template <typename PointType>
void KdTree<PointType>::searchByRange_(KdTreeNode *root, BoxPointType boxpoint, PointVector &storage)
{
    if (root == nullptr)
        return;
    pushDown_(root);
    if (boxpoint.vertex_max[0] <= root->node_range_x[0] || boxpoint.vertex_min[0] > root->node_range_x[1])
        return;
    if (boxpoint.vertex_max[1] <= root->node_range_y[0] || boxpoint.vertex_min[1] > root->node_range_y[1])
        return;
    if (boxpoint.vertex_max[2] <= root->node_range_z[0] || boxpoint.vertex_min[2] > root->node_range_z[1])
        return;
    if (boxpoint.vertex_min[0] <= root->node_range_x[0] && boxpoint.vertex_max[0] > root->node_range_x[1] && boxpoint.vertex_min[1] <= root->node_range_y[0] && boxpoint.vertex_max[1] > root->node_range_y[1] && boxpoint.vertex_min[2] <= root->node_range_z[0] && boxpoint.vertex_max[2] > root->node_range_z[1])
    {
        flatten(root, storage, DeleteStorage::NOT_RECORD);
        return;
    }
    if (boxpoint.vertex_min[0] <= root->point.x && boxpoint.vertex_max[0] > root->point.x && boxpoint.vertex_min[1] <= root->point.y && boxpoint.vertex_max[1] > root->point.y && boxpoint.vertex_min[2] <= root->point.z && boxpoint.vertex_max[2] > root->point.z)
    {
        if (!root->point_deleted)
            storage.push_back(root->point);
    }
    if ((rebuildPtr_ == nullptr) || root->left_son_ptr != *rebuildPtr_)
    {
        searchByRange_(root->left_son_ptr, boxpoint, storage);
    }
    else
    {
        pthread_mutex_lock(&searchFlagMutex_);
        searchByRange_(root->left_son_ptr, boxpoint, storage);
        pthread_mutex_unlock(&searchFlagMutex_);
    }
    if ((rebuildPtr_ == nullptr) || root->right_son_ptr != *rebuildPtr_)
    {
        searchByRange_(root->right_son_ptr, boxpoint, storage);
    }
    else
    {
        pthread_mutex_lock(&searchFlagMutex_);
        searchByRange_(root->right_son_ptr, boxpoint, storage);
        pthread_mutex_unlock(&searchFlagMutex_);
    }
    return;
}

template <typename PointType>
void KdTree<PointType>::searchByRadius_(KdTreeNode *root, PointType point, float radius, PointVector &storage)
{
    if (root == nullptr)
        return;
    pushDown_(root);
    PointType range_center;
    range_center.x = (root->node_range_x[0] + root->node_range_x[1]) * 0.5;
    range_center.y = (root->node_range_y[0] + root->node_range_y[1]) * 0.5;
    range_center.z = (root->node_range_z[0] + root->node_range_z[1]) * 0.5;
    float dist = std::sqrt(calcDist_(range_center, point));
    if (dist > radius + std::sqrt(root->radius_sq)) return;
    if (dist <= radius - std::sqrt(root->radius_sq)) 
    {
        flatten(root, storage, DeleteStorage::NOT_RECORD);
        return;
    }
    if (!root->point_deleted && calcDist_(root->point, point) <= radius * radius){
        storage.push_back(root->point);
    }
    if ((rebuildPtr_ == nullptr) || root->left_son_ptr != *rebuildPtr_)
    {
        searchByRadius_(root->left_son_ptr, point, radius, storage);
    }
    else
    {
        pthread_mutex_lock(&searchFlagMutex_);
        searchByRadius_(root->left_son_ptr, point, radius, storage);
        pthread_mutex_unlock(&searchFlagMutex_);
    }
    if ((rebuildPtr_ == nullptr) || root->right_son_ptr != *rebuildPtr_)
    {
        searchByRadius_(root->right_son_ptr, point, radius, storage);
    }
    else
    {
        pthread_mutex_lock(&searchFlagMutex_);
        searchByRadius_(root->right_son_ptr, point, radius, storage);
        pthread_mutex_unlock(&searchFlagMutex_);
    }    
    return;
}

template <typename PointType>
bool KdTree<PointType>::criterionCheck_(KdTreeNode *root)
{
    if (root->TreeSize <= MINIMAL_UNBALANCED_TREE_SIZE)
    {
        return false;
    }
    float balance_evaluation = 0.0f;
    float delete_evaluation = 0.0f;
    KdTreeNode *son_ptr = root->left_son_ptr;
    if (son_ptr == nullptr)
        son_ptr = root->right_son_ptr;
    delete_evaluation = float(root->invalid_point_num) / root->TreeSize;
    balance_evaluation = float(son_ptr->TreeSize) / (root->TreeSize - 1);
    if (delete_evaluation > deleteCriterionParam_)
    {
        return true;
    }
    if (balance_evaluation > balanceCriterionParam_ || balance_evaluation < 1 - balanceCriterionParam_)
    {
        return true;
    }
    return false;
}

template <typename PointType>
void KdTree<PointType>::pushDown_(KdTreeNode *root)
{
    if (root == nullptr)
        return;
    OperationLog operation;
    operation.op = Operation::PUSH_DOWN;
    operation.tree_deleted = root->tree_deleted;
    operation.tree_downsample_deleted = root->tree_downsample_deleted;
    if (root->need_push_down_to_left && root->left_son_ptr != nullptr)
    {
        if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->left_son_ptr)
        {
            root->left_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
            root->left_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
            root->left_son_ptr->tree_deleted = root->tree_deleted || root->left_son_ptr->tree_downsample_deleted;
            root->left_son_ptr->point_deleted = root->left_son_ptr->tree_deleted || root->left_son_ptr->point_downsample_deleted;
            if (root->tree_downsample_deleted)
                root->left_son_ptr->down_del_num = root->left_son_ptr->TreeSize;
            if (root->tree_deleted)
                root->left_son_ptr->invalid_point_num = root->left_son_ptr->TreeSize;
            else
                root->left_son_ptr->invalid_point_num = root->left_son_ptr->down_del_num;
            root->left_son_ptr->need_push_down_to_left = true;
            root->left_son_ptr->need_push_down_to_right = true;
            root->need_push_down_to_left = false;
        }
        else
        {
            pthread_mutex_lock(&workingFlagMutex_);
            root->left_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
            root->left_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
            root->left_son_ptr->tree_deleted = root->tree_deleted || root->left_son_ptr->tree_downsample_deleted;
            root->left_son_ptr->point_deleted = root->left_son_ptr->tree_deleted || root->left_son_ptr->point_downsample_deleted;
            if (root->tree_downsample_deleted)
                root->left_son_ptr->down_del_num = root->left_son_ptr->TreeSize;
            if (root->tree_deleted)
                root->left_son_ptr->invalid_point_num = root->left_son_ptr->TreeSize;
            else
                root->left_son_ptr->invalid_point_num = root->left_son_ptr->down_del_num;
            root->left_son_ptr->need_push_down_to_left = true;
            root->left_son_ptr->need_push_down_to_right = true;
            if (rebuildFlag_)
            {
                pthread_mutex_lock(&rebuildLoggerMutexLock_);
                rebuildLogger_.push(operation);
                pthread_mutex_unlock(&rebuildLoggerMutexLock_);
            }
            root->need_push_down_to_left = false;
            pthread_mutex_unlock(&workingFlagMutex_);
        }
    }
    if (root->need_push_down_to_right && root->right_son_ptr != nullptr)
    {
        if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->right_son_ptr)
        {
            root->right_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
            root->right_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
            root->right_son_ptr->tree_deleted = root->tree_deleted || root->right_son_ptr->tree_downsample_deleted;
            root->right_son_ptr->point_deleted = root->right_son_ptr->tree_deleted || root->right_son_ptr->point_downsample_deleted;
            if (root->tree_downsample_deleted)
                root->right_son_ptr->down_del_num = root->right_son_ptr->TreeSize;
            if (root->tree_deleted)
                root->right_son_ptr->invalid_point_num = root->right_son_ptr->TreeSize;
            else
                root->right_son_ptr->invalid_point_num = root->right_son_ptr->down_del_num;
            root->right_son_ptr->need_push_down_to_left = true;
            root->right_son_ptr->need_push_down_to_right = true;
            root->need_push_down_to_right = false;
        }
        else
        {
            pthread_mutex_lock(&workingFlagMutex_);
            root->right_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
            root->right_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
            root->right_son_ptr->tree_deleted = root->tree_deleted || root->right_son_ptr->tree_downsample_deleted;
            root->right_son_ptr->point_deleted = root->right_son_ptr->tree_deleted || root->right_son_ptr->point_downsample_deleted;
            if (root->tree_downsample_deleted)
                root->right_son_ptr->down_del_num = root->right_son_ptr->TreeSize;
            if (root->tree_deleted)
                root->right_son_ptr->invalid_point_num = root->right_son_ptr->TreeSize;
            else
                root->right_son_ptr->invalid_point_num = root->right_son_ptr->down_del_num;
            root->right_son_ptr->need_push_down_to_left = true;
            root->right_son_ptr->need_push_down_to_right = true;
            if (rebuildFlag_)
            {
                pthread_mutex_lock(&rebuildLoggerMutexLock_);
                rebuildLogger_.push(operation);
                pthread_mutex_unlock(&rebuildLoggerMutexLock_);
            }
            root->need_push_down_to_right = false;
            pthread_mutex_unlock(&workingFlagMutex_);
        }
    }
    return;
}

template <typename PointType>
void KdTree<PointType>::update_(KdTreeNode *root)
{
    KdTreeNode *left_son_ptr = root->left_son_ptr;
    KdTreeNode *right_son_ptr = root->right_son_ptr;
    float tmp_range_x[2] = {INFINITY, -INFINITY};
    float tmp_range_y[2] = {INFINITY, -INFINITY};
    float tmp_range_z[2] = {INFINITY, -INFINITY};
    // update_ Tree Size
    if (left_son_ptr != nullptr && right_son_ptr != nullptr)
    {
        root->TreeSize = left_son_ptr->TreeSize + right_son_ptr->TreeSize + 1;
        root->invalid_point_num = left_son_ptr->invalid_point_num + right_son_ptr->invalid_point_num + (root->point_deleted ? 1 : 0);
        root->down_del_num = left_son_ptr->down_del_num + right_son_ptr->down_del_num + (root->point_downsample_deleted ? 1 : 0);
        root->tree_downsample_deleted = left_son_ptr->tree_downsample_deleted & right_son_ptr->tree_downsample_deleted & root->point_downsample_deleted;
        root->tree_deleted = left_son_ptr->tree_deleted && right_son_ptr->tree_deleted && root->point_deleted;
        if (root->tree_deleted || (!left_son_ptr->tree_deleted && !right_son_ptr->tree_deleted && !root->point_deleted))
        {
            tmp_range_x[0] = std::min(std::min(left_son_ptr->node_range_x[0], right_son_ptr->node_range_x[0]), root->point.x);
            tmp_range_x[1] = std::max(std::max(left_son_ptr->node_range_x[1], right_son_ptr->node_range_x[1]), root->point.x);
            tmp_range_y[0] = std::min(std::min(left_son_ptr->node_range_y[0], right_son_ptr->node_range_y[0]), root->point.y);
            tmp_range_y[1] = std::max(std::max(left_son_ptr->node_range_y[1], right_son_ptr->node_range_y[1]), root->point.y);
            tmp_range_z[0] = std::min(std::min(left_son_ptr->node_range_z[0], right_son_ptr->node_range_z[0]), root->point.z);
            tmp_range_z[1] = std::max(std::max(left_son_ptr->node_range_z[1], right_son_ptr->node_range_z[1]), root->point.z);
        }
        else
        {
            if (!left_son_ptr->tree_deleted)
            {
                tmp_range_x[0] = std::min(tmp_range_x[0], left_son_ptr->node_range_x[0]);
                tmp_range_x[1] = std::max(tmp_range_x[1], left_son_ptr->node_range_x[1]);
                tmp_range_y[0] = std::min(tmp_range_y[0], left_son_ptr->node_range_y[0]);
                tmp_range_y[1] = std::max(tmp_range_y[1], left_son_ptr->node_range_y[1]);
                tmp_range_z[0] = std::min(tmp_range_z[0], left_son_ptr->node_range_z[0]);
                tmp_range_z[1] = std::max(tmp_range_z[1], left_son_ptr->node_range_z[1]);
            }
            if (!right_son_ptr->tree_deleted)
            {
                tmp_range_x[0] = std::min(tmp_range_x[0], right_son_ptr->node_range_x[0]);
                tmp_range_x[1] = std::max(tmp_range_x[1], right_son_ptr->node_range_x[1]);
                tmp_range_y[0] = std::min(tmp_range_y[0], right_son_ptr->node_range_y[0]);
                tmp_range_y[1] = std::max(tmp_range_y[1], right_son_ptr->node_range_y[1]);
                tmp_range_z[0] = std::min(tmp_range_z[0], right_son_ptr->node_range_z[0]);
                tmp_range_z[1] = std::max(tmp_range_z[1], right_son_ptr->node_range_z[1]);
            }
            if (!root->point_deleted)
            {
                tmp_range_x[0] = std::min(tmp_range_x[0], root->point.x);
                tmp_range_x[1] = std::max(tmp_range_x[1], root->point.x);
                tmp_range_y[0] = std::min(tmp_range_y[0], root->point.y);
                tmp_range_y[1] = std::max(tmp_range_y[1], root->point.y);
                tmp_range_z[0] = std::min(tmp_range_z[0], root->point.z);
                tmp_range_z[1] = std::max(tmp_range_z[1], root->point.z);
            }
        }
    }
    else if (left_son_ptr != nullptr)
    {
        root->TreeSize = left_son_ptr->TreeSize + 1;
        root->invalid_point_num = left_son_ptr->invalid_point_num + (root->point_deleted ? 1 : 0);
        root->down_del_num = left_son_ptr->down_del_num + (root->point_downsample_deleted ? 1 : 0);
        root->tree_downsample_deleted = left_son_ptr->tree_downsample_deleted & root->point_downsample_deleted;
        root->tree_deleted = left_son_ptr->tree_deleted && root->point_deleted;
        if (root->tree_deleted || (!left_son_ptr->tree_deleted && !root->point_deleted))
        {
            tmp_range_x[0] = std::min(left_son_ptr->node_range_x[0], root->point.x);
            tmp_range_x[1] = std::max(left_son_ptr->node_range_x[1], root->point.x);
            tmp_range_y[0] = std::min(left_son_ptr->node_range_y[0], root->point.y);
            tmp_range_y[1] = std::max(left_son_ptr->node_range_y[1], root->point.y);
            tmp_range_z[0] = std::min(left_son_ptr->node_range_z[0], root->point.z);
            tmp_range_z[1] = std::max(left_son_ptr->node_range_z[1], root->point.z);
        }
        else
        {
            if (!left_son_ptr->tree_deleted)
            {
                tmp_range_x[0] = std::min(tmp_range_x[0], left_son_ptr->node_range_x[0]);
                tmp_range_x[1] = std::max(tmp_range_x[1], left_son_ptr->node_range_x[1]);
                tmp_range_y[0] = std::min(tmp_range_y[0], left_son_ptr->node_range_y[0]);
                tmp_range_y[1] = std::max(tmp_range_y[1], left_son_ptr->node_range_y[1]);
                tmp_range_z[0] = std::min(tmp_range_z[0], left_son_ptr->node_range_z[0]);
                tmp_range_z[1] = std::max(tmp_range_z[1], left_son_ptr->node_range_z[1]);
            }
            if (!root->point_deleted)
            {
                tmp_range_x[0] = std::min(tmp_range_x[0], root->point.x);
                tmp_range_x[1] = std::max(tmp_range_x[1], root->point.x);
                tmp_range_y[0] = std::min(tmp_range_y[0], root->point.y);
                tmp_range_y[1] = std::max(tmp_range_y[1], root->point.y);
                tmp_range_z[0] = std::min(tmp_range_z[0], root->point.z);
                tmp_range_z[1] = std::max(tmp_range_z[1], root->point.z);
            }
        }
    }
    else if (right_son_ptr != nullptr)
    {
        root->TreeSize = right_son_ptr->TreeSize + 1;
        root->invalid_point_num = right_son_ptr->invalid_point_num + (root->point_deleted ? 1 : 0);
        root->down_del_num = right_son_ptr->down_del_num + (root->point_downsample_deleted ? 1 : 0);
        root->tree_downsample_deleted = right_son_ptr->tree_downsample_deleted & root->point_downsample_deleted;
        root->tree_deleted = right_son_ptr->tree_deleted && root->point_deleted;
        if (root->tree_deleted || (!right_son_ptr->tree_deleted && !root->point_deleted))
        {
            tmp_range_x[0] = std::min(right_son_ptr->node_range_x[0], root->point.x);
            tmp_range_x[1] = std::max(right_son_ptr->node_range_x[1], root->point.x);
            tmp_range_y[0] = std::min(right_son_ptr->node_range_y[0], root->point.y);
            tmp_range_y[1] = std::max(right_son_ptr->node_range_y[1], root->point.y);
            tmp_range_z[0] = std::min(right_son_ptr->node_range_z[0], root->point.z);
            tmp_range_z[1] = std::max(right_son_ptr->node_range_z[1], root->point.z);
        }
        else
        {
            if (!right_son_ptr->tree_deleted)
            {
                tmp_range_x[0] = std::min(tmp_range_x[0], right_son_ptr->node_range_x[0]);
                tmp_range_x[1] = std::max(tmp_range_x[1], right_son_ptr->node_range_x[1]);
                tmp_range_y[0] = std::min(tmp_range_y[0], right_son_ptr->node_range_y[0]);
                tmp_range_y[1] = std::max(tmp_range_y[1], right_son_ptr->node_range_y[1]);
                tmp_range_z[0] = std::min(tmp_range_z[0], right_son_ptr->node_range_z[0]);
                tmp_range_z[1] = std::max(tmp_range_z[1], right_son_ptr->node_range_z[1]);
            }
            if (!root->point_deleted)
            {
                tmp_range_x[0] = std::min(tmp_range_x[0], root->point.x);
                tmp_range_x[1] = std::max(tmp_range_x[1], root->point.x);
                tmp_range_y[0] = std::min(tmp_range_y[0], root->point.y);
                tmp_range_y[1] = std::max(tmp_range_y[1], root->point.y);
                tmp_range_z[0] = std::min(tmp_range_z[0], root->point.z);
                tmp_range_z[1] = std::max(tmp_range_z[1], root->point.z);
            }
        }
    }
    else
    {
        root->TreeSize = 1;
        root->invalid_point_num = (root->point_deleted ? 1 : 0);
        root->down_del_num = (root->point_downsample_deleted ? 1 : 0);
        root->tree_downsample_deleted = root->point_downsample_deleted;
        root->tree_deleted = root->point_deleted;
        tmp_range_x[0] = root->point.x;
        tmp_range_x[1] = root->point.x;
        tmp_range_y[0] = root->point.y;
        tmp_range_y[1] = root->point.y;
        tmp_range_z[0] = root->point.z;
        tmp_range_z[1] = root->point.z;
    }
    std::memcpy(root->node_range_x, tmp_range_x, sizeof(tmp_range_x));
    std::memcpy(root->node_range_y, tmp_range_y, sizeof(tmp_range_y));
    std::memcpy(root->node_range_z, tmp_range_z, sizeof(tmp_range_z));
    float x_L = (root->node_range_x[1] - root->node_range_x[0]) * 0.5;
    float y_L = (root->node_range_y[1] - root->node_range_y[0]) * 0.5;
    float z_L = (root->node_range_z[1] - root->node_range_z[0]) * 0.5;
    root->radius_sq = x_L*x_L + y_L * y_L + z_L * z_L;
    if (left_son_ptr != nullptr)
        left_son_ptr->father_ptr = root;
    if (right_son_ptr != nullptr)
        right_son_ptr->father_ptr = root;
    if (root == root_ && root->TreeSize > 3)
    {
        KdTreeNode *son_ptr = root->left_son_ptr;
        if (son_ptr == nullptr)
            son_ptr = root->right_son_ptr;
        float tmp_bal = float(son_ptr->TreeSize) / (root->TreeSize - 1);
        root->alpha_del = float(root->invalid_point_num) / root->TreeSize;
        root->alpha_bal = (tmp_bal >= 0.5 - EPS) ? tmp_bal : 1 - tmp_bal;
    }
    return;
}

template <typename PointType>
void KdTree<PointType>::flatten(KdTreeNode *root, PointVector &storage, DeleteStorage storageType)
{
    if (root == nullptr)
        return;
    pushDown_(root);
    if (!root->point_deleted)
    {
        storage.push_back(root->point);
    }
    flatten(root->left_son_ptr, storage, storageType);
    flatten(root->right_son_ptr, storage, storageType);
    switch (storageType)
    {
    case DeleteStorage::NOT_RECORD:
        break;
    case DeleteStorage::DELETE_POINTS_REC:
        if (root->point_deleted && !root->point_downsample_deleted)
        {
            pointsDeleted_.push_back(root->point);
        }
        break;
    case DeleteStorage::MULTI_THREAD_REC:
        if (root->point_deleted && !root->point_downsample_deleted)
        {
            multithreadPointsDeleted_.push_back(root->point);
        }
        break;
    default:
        break;
    }
    return;
}

template <typename PointType>
void KdTree<PointType>::deleteTreeNodes_(KdTreeNode **root)
{
    if (*root == nullptr)
        return;
    pushDown_(*root);
    deleteTreeNodes_(&(*root)->left_son_ptr);
    deleteTreeNodes_(&(*root)->right_son_ptr);

    pthread_mutex_destroy(&(*root)->push_down_mutex_lock);
    delete *root;
    *root = nullptr;

    return;
}

template <typename PointType>
bool KdTree<PointType>::samePoint_(PointType a, PointType b)
{
    return (std::fabs(a.x - b.x) < EPS && std::fabs(a.y - b.y) < EPS && std::fabs(a.z - b.z) < EPS);
}

template <typename PointType>
float KdTree<PointType>::calcDist_(PointType a, PointType b)
{
    float dist = 0.0f;
    dist = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
    return dist;
}

template <typename PointType>
float KdTree<PointType>::calcBoxDist_(KdTreeNode *node, PointType point)
{
    if (node == nullptr)
        return INFINITY;
    float min_dist = 0.0;
    if (point.x < node->node_range_x[0])
        min_dist += (point.x - node->node_range_x[0]) * (point.x - node->node_range_x[0]);
    if (point.x > node->node_range_x[1])
        min_dist += (point.x - node->node_range_x[1]) * (point.x - node->node_range_x[1]);
    if (point.y < node->node_range_y[0])
        min_dist += (point.y - node->node_range_y[0]) * (point.y - node->node_range_y[0]);
    if (point.y > node->node_range_y[1])
        min_dist += (point.y - node->node_range_y[1]) * (point.y - node->node_range_y[1]);
    if (point.z < node->node_range_z[0])
        min_dist += (point.z - node->node_range_z[0]) * (point.z - node->node_range_z[0]);
    if (point.z > node->node_range_z[1])
        min_dist += (point.z - node->node_range_z[1]) * (point.z - node->node_range_z[1]);
    return min_dist;
}

template <typename PointType>
bool KdTree<PointType>::pointCmpX_(PointType a, PointType b) { return a.x < b.x; }
template <typename PointType>
bool KdTree<PointType>::pointCmpY_(PointType a, PointType b) { return a.y < b.y; }
template <typename PointType>
bool KdTree<PointType>::pointCmpZ_(PointType a, PointType b) { return a.z < b.z; }

} // arc

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
