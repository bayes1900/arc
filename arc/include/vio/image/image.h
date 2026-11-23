#pragma once

#include <base/base.h>

#define ARC_HOST_DEVICE
#define ARC_EXTENSION_IMAGE

namespace arc {

template <typename T>
struct CopyObject {
    CopyObject(const T& obj) : obj(obj) {}
    const T& obj;
};
inline void pitchedCopy(
    char* dst,
    unsigned int dstPitchBytes,
    const char* src,
    unsigned int srcPitchBytes,
    unsigned int widthBytes,
    unsigned int height
) {
    if (dstPitchBytes == widthBytes && srcPitchBytes == widthBytes) {
        std::memcpy(dst, src, height * widthBytes);
    } else {
        for (unsigned int row = 0; row < height; ++ row) {
            std::memcpy(dst, src, widthBytes);
            dst += dstPitchBytes;
            src += srcPitchBytes;
        }
    }
}
inline void cubHermiteSpline(
    const double& p0, const double& p1, const double& p2, const double& p3, const double x, double* f, double* dfdx) {
    const double a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
    const double b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
    const double c = 0.5 * (-p0 + p2);
    const double d = p1;
    if (f != NULL) {
        *f = d + x * (c + x * (b + x * a));
    }
    if (dfdx != NULL) {
        *dfdx = c + x * (2.0 * b + 3.0 * a * x);
    }
}

template <typename T>
struct Image {
    using PixelType = T;

    Image()
        : pitch(0)
        , ptr(0)
        , w(0)
        , h(0)
    {}

    Image(T* ptr, size_t w, size_t h, size_t pitch)
        : pitch(pitch)
        , ptr(ptr)
        , w(w)
        , h(h)
    {}

    ARC_HOST_DEVICE inline size_t sizeBytes() const { return pitch * h; }
    ARC_HOST_DEVICE inline size_t area() const { return w * h; }
    ARC_HOST_DEVICE inline bool isValid() const { return ptr != 0; }
    ARC_HOST_DEVICE inline bool isContiguous() const { return w * sizeof(T) == pitch; }
    ARC_HOST_DEVICE inline T* begin() { return ptr; }
    ARC_HOST_DEVICE inline T* end() { return rowPtr(h - 1) + w; }
    ARC_HOST_DEVICE inline const T* begin() const { return ptr; }
    ARC_HOST_DEVICE inline const T* end() const { return rowPtr(h - 1) + w; }
    ARC_HOST_DEVICE inline size_t size() const { return w * h; }

    template <typename UnaryOperation>
    ARC_HOST_DEVICE inline void transform(UnaryOperation unaryOp) {
        ensure(isValid(), "VIO check failed");
        for (size_t y = 0; y < h; ++ y) {
            T* el = rowPtr(y);
            const T* elEnd = el + w;
            for (; el != elEnd; ++ el) {
                *el = unaryOp(*el);
            }
        }
    }
    ARC_HOST_DEVICE inline void fill(const T& val) {
        transform([&](const T&) { return val; });
    }
    ARC_HOST_DEVICE inline void replace(const T& oldval, const T& newval) {
        Transform([&](const T& val) { return (val == oldval) ? newval : val; });
    }
    void memset(unsigned char v = 0) {
        ensure(isValid(), "VIO check failed");
        if (isContiguous()) {
            std::memset((char*)ptr, v, pitch * h);
        } else {
            for (size_t y = 0; y < h; ++ y) {
                std::memset((char*)rowPtr(y), v, pitch);
            }
        }
    }
    void copyFrom(const Image<T>& img) {
        if (isValid() && img.isValid()) {
            ensure(w >= img.w && h >= img.h, "VIO check failed");
            pitchedCopy(
                (char*)ptr, pitch, (char*)img.ptr, img.pitch, std::min(img.w, w) * sizeof(T), std::min(img.h, h));
        } else if (img.isValid() != isValid()) {
            ensure(false && "Cannot copy from / to an unasigned image.", "VIO check failed");
        }
    }

    template <typename BinaryOperation>
    ARC_HOST_DEVICE inline T accumulate(const T init, BinaryOperation binaryOp) {
        ensure(isValid(), "VIO check failed");
        T val = init;
        for (size_t y = 0; y < h; ++ y) {
            T* el = rowPtr(y);
            const T* elEnd = el + w;
            for (; el != elEnd; ++ el) {
                val = binaryOp(val, *el);
            }
        }
        return val;
    }
    std::pair<T, T> minMax() const {
        ensure(isValid(), "VIO check failed");
        std::pair<T, T> minmax(std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest());
        for (size_t r = 0; r < h; ++ r) {
            const T* ptr = rowPtr(r);
            const T* end = ptr + w;
            while (ptr != end) {
                minmax.first = std::min(*ptr, minmax.first);
                minmax.second = std::max(*ptr, minmax.second);
                ++ ptr;
            }
        }
        return minmax;
    }

    template <typename Tout = T>
    Tout sum() const {
        return Accumulate((T)0, [](const T& lhs, const T& rhs) { return lhs + rhs; });
    }

    template <typename Tout = T>
    Tout mean() const {
        return sum<Tout>() / area();
    }
    ARC_HOST_DEVICE inline T* rowPtr(size_t y) { return (T*)((unsigned char*)(ptr) + y * pitch); }
    ARC_HOST_DEVICE inline const T* rowPtr(size_t y) const { return (T*)((unsigned char*)(ptr) + y * pitch); }
    ARC_HOST_DEVICE inline T& operator()(size_t x, size_t y) {
        return rowPtr(y)[x];
    }
    ARC_HOST_DEVICE inline const T& operator()(size_t x, size_t y) const {
        return rowPtr(y)[x];
    }

    template <typename TVec>
    ARC_HOST_DEVICE inline T& operator()(const TVec& p) {
        return rowPtr(p[1])[p[0]];
    }

    template <typename TVec>
    ARC_HOST_DEVICE inline const T& operator()(const TVec& p) const {
        return rowPtr(p[1])[p[0]];
    }

    ARC_HOST_DEVICE inline T& operator[](size_t ix) {
        return ptr[ix];
    }

    ARC_HOST_DEVICE inline const T& operator[](size_t ix) const {
        return ptr[ix];
    }

    template <typename S>
    S interp(const Eigen::Matrix<S, 2, 1>& p) const {
        return interp<S>(p[0], p[1]);
    }

    template <typename S>
    Eigen::Matrix<S, 3, 1> interpGrad(const Eigen::Matrix<S, 2, 1>& p) const {
        return interpGrad<S>(p[0], p[1]);
    }

    template <typename S>
    Eigen::Matrix<S, 3, 1> interpGradBilinearExact(const Eigen::Matrix<S, 2, 1>& p) const {
        return interpGradBilinearExact<S>(p[0], p[1]);
    }

    template <typename S>
    double interpCubicSplines(const Eigen::Matrix<S, 2, 1>& p) const {
        return interpCubicSplines<S>(p[0], p[1]);
    }

    template <typename S>
    Eigen::Matrix<S, 3, 1> interpGradCubicSplines(const Eigen::Matrix<S, 2, 1>& p) const {
        return interpGradCubicSplines<S>(p[0], p[1]);
    }
    void clamp(int& ixm1, int& ixp1, int& ixp2, int& iym1, int& iyp1, int& iyp2) const {
        if (iym1 < 0)
            iym1 = 0;
        if (ixm1 < 0)
            ixm1 = 0;
        if (ixp1 > (int)w - 1)
            ixp1 = w - 1;
        if (ixp2 > (int)w - 1)
            ixp2 = w - 1;
        if (iyp1 > (int)h - 1)
            iyp1 = h - 1;
        if (iyp2 > (int)h - 1)
            iyp2 = h - 1;
    }

    template <typename S>
    S interp(S x, S y) const {
        static_assert(std::is_floating_point_v<S>,
            "interpolation / gradient only makes sense "
            "for floating point result type");
        int ix = x;
        int iy = y;
        S dx = x - ix;
        S dy = y - iy;
        S ddx = S(1.0) - dx;
        S ddy = S(1.0) - dy;
        return ddx * ddy * (*this)(ix, iy) + ddx * dy * (*this)(ix, iy + 1) + dx * ddy * (*this)(ix + 1, iy) +
            dx * dy * (*this)(ix + 1, iy + 1);
    }

    template <typename S>
    Eigen::Matrix<S, 3, 1> interpGrad(S x, S y) const {
        static_assert(std::is_floating_point_v<S>,
            "interpolation / gradient only makes sense "
            "for floating point result type");
        int ix = x;
        int iy = y;
        S dx = x - ix;
        S dy = y - iy;
        S ddx = S(1.0) - dx;
        S ddy = S(1.0) - dy;
        Eigen::Matrix<S, 3, 1> res;
        const T& px0y0 = (*this)(ix, iy);
        const T& px1y0 = (*this)(ix + 1, iy);
        const T& px0y1 = (*this)(ix, iy + 1);
        const T& px1y1 = (*this)(ix + 1, iy + 1);
        res[0] = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 + dx * dy * px1y1;
        const T& pxm1y0 = (*this)(ix - 1, iy);
        const T& pxm1y1 = (*this)(ix - 1, iy + 1);
        S resMx = ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 + dx * dy * px0y1;
        const T& px2y0 = (*this)(ix + 2, iy);
        const T& px2y1 = (*this)(ix + 2, iy + 1);
        S resPx = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 + dx * dy * px2y1;
        res[1] = S(0.5) * (resPx - resMx);
        const T& px0ym1 = (*this)(ix, iy - 1);
        const T& px1ym1 = (*this)(ix + 1, iy - 1);
        S resMy = ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 + dx * dy * px1y0;
        const T& px0y2 = (*this)(ix, iy + 2);
        const T& px1y2 = (*this)(ix + 1, iy + 2);
        S resPy = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 + dx * dy * px1y2;
        res[2] = S(0.5) * (resPy - resMy);
        return res;
    }

    template <typename S>
    double interpCubicSplines(S x, S y) const {
        static_assert(std::is_floating_point_v<S>,
            "interpolation / gradient only makes sense "
            "for floating point result type");
        double imageValue;
        double p0, p1, p2, p3;
        double f0, f1, f2, f3;
        int ix = x;
        int iy = y;
        int ixm1 = ix - 1;
        int ixp1 = ix + 1;
        int ixp2 = ix + 2;
        int iym1 = iy - 1;
        int iyp1 = iy + 1;
        int iyp2 = iy + 2;
        clamp(ixm1, ixp1, ixp2, iym1, iyp1, iyp2);
        p0 = (*this)(ixm1, iym1);
        p1 = (*this)(ix, iym1);
        p2 = (*this)(ixp1, iym1);
        p3 = (*this)(ixp2, iym1);
        CubHermiteSpline(p0, p1, p2, p3, x - ix, &f0, NULL);
        p0 = (*this)(ixm1, iy);
        p1 = (*this)(ix, iy);
        p2 = (*this)(ixp1, iy);
        p3 = (*this)(ixp2, iy);
        CubHermiteSpline(p0, p1, p2, p3, x - ix, &f1, NULL);
        p0 = (*this)(ixm1, iyp1);
        p1 = (*this)(ix, iyp1);
        p2 = (*this)(ixp1, iyp1);
        p3 = (*this)(ixp2, iyp1);
        CubHermiteSpline(p0, p1, p2, p3, x - ix, &f2, NULL);
        p0 = (*this)(ixm1, iyp2);
        p1 = (*this)(ix, iyp2);
        p2 = (*this)(ixp1, iyp2);
        p3 = (*this)(ixp2, iyp2);
        CubHermiteSpline(p0, p1, p2, p3, x - ix, &f3, NULL);
        CubHermiteSpline(f0, f1, f2, f3, y - iy, &imageValue, NULL);
        return imageValue;
    }

    template <typename S>
    Eigen::Matrix<S, 3, 1> interpGradCubicSplines(S x, S y) const {
        static_assert(std::is_floating_point_v<S>,
            "interpolation / gradient only makes sense "
            "for floating point result type");
        Eigen::Matrix<S, 3, 1> res;
        int ix = x;
        int iy = y;
        double p0, p1, p2, p3;
        double f0, f1, f2, f3;
        double df0dx, df1dx, df2dx, df3dx;
        int ixm1 = ix - 1;
        int ixp1 = ix + 1;
        int ixp2 = ix + 2;
        int iym1 = iy - 1;
        int iyp1 = iy + 1;
        int iyp2 = iy + 2;
        clamp(ixm1, ixp1, ixp2, iym1, iyp1, iyp2);
        p0 = (*this)(ixm1, iym1);
        p1 = (*this)(ix, iym1);
        p2 = (*this)(ixp1, iym1);
        p3 = (*this)(ixp2, iym1);
        CubHermiteSpline(p0, p1, p2, p3, x - ix, &f0, &df0dx);
        p0 = (*this)(ixm1, iy);
        p1 = (*this)(ix, iy);
        p2 = (*this)(ixp1, iy);
        p3 = (*this)(ixp2, iy);
        CubHermiteSpline(p0, p1, p2, p3, x - ix, &f1, &df1dx);
        p0 = (*this)(ixm1, iyp1);
        p1 = (*this)(ix, iyp1);
        p2 = (*this)(ixp1, iyp1);
        p3 = (*this)(ixp2, iyp1);
        CubHermiteSpline(p0, p1, p2, p3, x - ix, &f2, &df2dx);
        p0 = (*this)(ixm1, iyp2);
        p1 = (*this)(ix, iyp2);
        p2 = (*this)(ixp1, iyp2);
        p3 = (*this)(ixp2, iyp2);
        CubHermiteSpline(p0, p1, p2, p3, x - ix, &f3, &df3dx);
        CubHermiteSpline(f0, f1, f2, f3, y - iy, &res[0], &res[2]);
        CubHermiteSpline(df0dx, df1dx, df2dx, df3dx, y - iy, &res[1], NULL);
        return res;
    }

    template <typename S>
    Eigen::Matrix<S, 3, 1> interpGradBilinearExact(S x, S y) const {
        static_assert(std::is_floating_point_v<S>,
            "interpolation / gradient only makes sense "
            "for floating point result type");
        Eigen::Matrix<S, 3, 1> res;
        int ix = x;
        int iy = y;
        S dx = x - ix;
        S dy = y - iy;
        S ddx = 1.0f - dx;
        S ddy = 1.0f - dy;
        int ixp1 = ix + 1;
        int iyp1 = iy + 1;
        if (ixp1 > (int)w - 1)
            ixp1 = w - 1;
        if (iyp1 > (int)h - 1)
            iyp1 = h - 1;
        const T& px0y0 = (*this)(ix, iy);
        const T& px1y0 = (*this)(ixp1, iy);
        const T& px0y1 = (*this)(ix, iyp1);
        const T& px1y1 = (*this)(ixp1, iyp1);
        const double fxy0 = ddx * px0y0 + dx * px1y0;
        const double fxy1 = ddx * px0y1 + dx * px1y1;
        res[0] = fxy0 * ddy + fxy1 * dy;
        res[2] = fxy1 - fxy0;
        const double fx0y = ddy * px0y0 + dy * px0y1;
        const double fx1y = ddy * px1y0 + dy * px1y1;
        res[1] = fx1y - fx0y;
        return res;
    }

    ARC_HOST_DEVICE
    bool inImage(const T* ptest) const { return ptr <= ptest && ptest < rowPtr(h); }
    ARC_HOST_DEVICE inline bool inBounds(int x, int y) const { return 0 <= x && x < (int)w && 0 <= y && y < (int)h; }
    ARC_HOST_DEVICE inline bool inBounds(float x, float y, float border) const {
        return border <= x && x < (w - border - 1) && border <= y && y < (h - border - 1);
    }

    template <typename Derived>
    ARC_HOST_DEVICE inline bool inBounds(
        const Eigen::MatrixBase<Derived>& p, const typename Derived::Scalar border) const {
        static_assert(Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime == 2,
            "Derived must be a vector of size 2");
        using Scalar = typename Derived::Scalar;
        Scalar offset(0);
        if constexpr (std::is_floating_point_v<Scalar>) {
            offset = Scalar(1);
        }
        return border <= p[0] && p[0] < ((int)w - border - offset) && border <= p[1] &&
            p[1] < ((int)h - border - offset);
    }
    ARC_HOST_DEVICE inline const Image<const T> subImage(size_t x, size_t y, size_t width, size_t height) const {
        ensure((x + width) <= w && (y + height) <= h, "VIO check failed");
        return Image<const T>(rowPtr(y) + x, width, height, pitch);
    }
    ARC_HOST_DEVICE inline Image<T> subImage(size_t x, size_t y, size_t width, size_t height) {
        ensure((x + width) <= w && (y + height) <= h, "VIO check failed");
        return Image<T>(rowPtr(y) + x, width, height, pitch);
    }
    ARC_HOST_DEVICE inline Image<T> row(int y) const { return subImage(0, y, w, 1); }
    ARC_HOST_DEVICE inline Image<T> col(int x) const { return subImage(x, 0, 1, h); }

    template <typename TRecast>
    ARC_HOST_DEVICE inline Image<TRecast> reinterpret() const {
        ensure(sizeof(TRecast) == sizeof(T), "sizeof(TRecast) must match sizeof(T)");
        return unsafeReinterpret<TRecast>();
    }

    template <typename TRecast>
    ARC_HOST_DEVICE inline Image<TRecast> unsafeReinterpret() const {
        return Image<TRecast>((TRecast*)ptr, w, h, pitch);
    }

    Image(size_t w, size_t h, size_t pitch, T* ptr)
        : pitch(pitch)
        , ptr(ptr)
        , w(w)
        , h(h)
    {}

    void dealloc() {
        if (ptr) {
            ::operator delete(ptr);
            ptr = nullptr;
        }
    }
    void alloc(size_t w, size_t h, size_t pitch) {
        dealloc();
        this->w = w;
        this->h = h;
        this->pitch = pitch;
        this->ptr = (T*)::operator new(h* pitch);
    }

    size_t pitch;
    T* ptr;
    size_t w;
    size_t h;

    ARC_EXTENSION_IMAGE
};

template <class T>
using DefaultImageAllocator = std::allocator<T>;

template <typename T, class Allocator = DefaultImageAllocator<T>>
class ManagedImage : public Image<T> {
public:
    using PixelType = T;
    using Ptr = std::shared_ptr<ManagedImage<T, Allocator>>;
    ~ManagedImage() { deallocate(); }
    ManagedImage() {}

    ManagedImage(size_t w)
        : Image<T>(Allocator().allocate(w), w, 1, w * sizeof(T)) {}

    ManagedImage(size_t w, size_t h)
        : Image<T>(Allocator().allocate(w * h), w, h, w * sizeof(T)) {}

    ManagedImage(size_t w, size_t h, size_t pitchBytes)
        : Image<T>(Allocator().allocate((h * pitchBytes) / sizeof(T) + 1), w, h, pitchBytes) {}
    ManagedImage(const ManagedImage<T>& other) = delete;
    ManagedImage(ManagedImage<T, Allocator>&& img) { *this = std::move(img); }

    void operator=(ManagedImage<T, Allocator>&& img) {
        deallocate();
        Image<T>::pitch = img.pitch;
        Image<T>::ptr = img.ptr;
        Image<T>::w = img.w;
        Image<T>::h = img.h;
        img.ptr = nullptr;
    }

    template <typename TOther>
    ManagedImage(const CopyObject<TOther>& other) {
        copyFrom(other.obj);
    }

    template <typename TOther>
    void operator=(const CopyObject<TOther>& other) {
        copyFrom(other.obj);
    }
    void swap(ManagedImage<T>& img) {
        std::swap(img.pitch, Image<T>::pitch);
        std::swap(img.ptr, Image<T>::ptr);
        std::swap(img.w, Image<T>::w);
        std::swap(img.h, Image<T>::h);
    }
    void copyFrom(const Image<T>& img) {
        if (!Image<T>::IsValid() || Image<T>::w != img.w || Image<T>::h != img.h) {
            Reinitialise(img.w, img.h);
        }
        Image<T>::copyFrom(img);
    }
    void reinitialise(size_t w, size_t h) {
        if (!Image<T>::ptr || Image<T>::w != w || Image<T>::h != h) {
            *this = ManagedImage<T, Allocator>(w, h);
        }
    }
    void reinitialise(size_t w, size_t h, size_t pitch) {
        if (!Image<T>::ptr || Image<T>::w != w || Image<T>::h != h || Image<T>::pitch != pitch) {
            *this = ManagedImage<T, Allocator>(w, h, pitch);
        }
    }
    void deallocate() {
        if (Image<T>::ptr) {
            Allocator().deallocate(Image<T>::ptr, (Image<T>::h * Image<T>::pitch) / sizeof(T));
            Image<T>::ptr = nullptr;
        }
    }

    template <typename TOther, typename AllocOther>
    void ownAndReinterpret(ManagedImage<TOther, AllocOther>&& img) {
        deallocate();
        Image<T>::pitch = img.pitch;
        Image<T>::ptr = (T*)img.ptr;
        Image<T>::w = img.w;
        Image<T>::h = img.h;
        img.ptr = nullptr;
    }

    template <typename T1>
    void convertFrom(const ManagedImage<T1>& img) {
        Reinitialise(img.w, img.h);
        for (size_t j = 0; j < img.h; ++ j) {
            T* thisRow = this->rowPtr(j);
            const T1* otherRow = img.rowPtr(j);
            for (size_t i = 0; i < img.w; ++ i) {
                thisRow[i] = T(otherRow[i]);
            }
        }
    }

    void operator-=(const ManagedImage<T>& img) {
        for (size_t j = 0; j < img.h; ++ j) {
            T* thisRow = this->rowPtr(j);
            const T* otherRow = img.rowPtr(j);
            for (size_t i = 0; i < img.w; ++ i) {
                thisRow[i] -= otherRow[i];
            }
        }
    }
};

struct ImageData {
    ImageData()
        : exposure(0) {}

    ManagedImage<uint16_t>::Ptr img;
    double exposure;
};

} // arc
