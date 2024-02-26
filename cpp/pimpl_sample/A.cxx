#include "A.h"
#include <iostream>
using namespace std;

class A::Impl {
    public:
        Impl(int _a, int _b): a(_a), b(_b) {
            cout << a << ";" << b << endl;
        }
    private:
        int a;
        int b;
};

A::A() {
    m_impl = new Impl(1, 2);
}

A::~A() {
    delete m_impl;
}
