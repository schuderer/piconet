#include <iostream>
#include "gtest/gtest.h"
#include "piconet/nn.h"

using std::cout, std::endl;
using namespace ajs;

TEST(Bla, Sum)
{
    LOG("Start of test code");

//    NN nn = NN();
//    nn.bla();

    LOG("End of test code");
    EXPECT_EQ(2, 1 + 1);
}
