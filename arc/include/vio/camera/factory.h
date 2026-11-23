#pragma once

#include <vio/camera/base.h>

namespace arc {

template <typename Scalar>
GenericCamera<Scalar> GenericCamera<Scalar>::fromString(const std::string& name) {
    GenericCamera<Scalar> res;
    constexpr size_t VARIANT_SIZE = std::variant_size<VariantT>::value;
    visitAllTypes_<VARIANT_SIZE - 1>(res, name);
    return res;
}

} // arc
