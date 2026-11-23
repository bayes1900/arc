#include <gtest/gtest.h>

#include <lio/lio.h>

int lioOdrValue();

TEST(LioOdr, LinkTwoTranslationUnits) {
    arc::LioConfig config;
    EXPECT_EQ(config.scanLine, 16);
    EXPECT_EQ(lioOdrValue(), 16);
}
