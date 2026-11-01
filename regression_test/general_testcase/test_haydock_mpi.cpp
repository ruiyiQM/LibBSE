// Author: Ruiyi Zhou.
#include "bse/haydock.h"
#include "utils/memory.h"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using Complex = std::complex<double>;
using Vector = std::vector<Complex>;

void check(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}

template<class F> void rejected(F run, MPI_Comm comm) {
    int caught = 0;
    try { run(); } catch (const std::exception &) { caught = 1; }
    MPI_Allreduce(MPI_IN_PLACE, &caught, 1, MPI_INT, MPI_MIN, comm);
    check(caught, "invalid distributed input did not fail on every rank");
}

void exercise(int n, int steps, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
    std::vector<int> counts(size), offsets(size);
    for (int p=0; p<size; ++p) {
        counts[p] = n/size + (p<n%size);
        offsets[p] = p*(n/size) + std::min(p,n%size);
    }
    Vector seed(n);
    for (int i=0; i<n; ++i) seed[i] = std::polar(1.+0.1*std::sin(i),0.37*i);
    const auto matrix = [n](const Vector &x, Vector &y) {
        y.resize(n);
        for (int i=0; i<n; ++i) {
            y[i] = (2.+0.001*i)*x[i];
            if (i) y[i] += Complex(0.03,-0.02)*x[i-1];
            if (i+1<n) y[i] += Complex(0.03,0.02)*x[i+1];
        }
    };
    std::uint64_t reference_bytes=0, local_bytes=0;
    const auto reference = libbse::haydock_recursion([&](const auto &x, auto &y) {
        reference_bytes=std::max(reference_bytes,libbse::MemoryTracker::instance().snapshot().current);
        matrix(x,y);
    },seed,steps);
    Vector local_seed(seed.begin()+offsets[rank],seed.begin()+offsets[rank]+counts[rank]);
    Vector all_x(n), all_y;
    const libbse::TdaApply local_apply = [&](const Vector &x, Vector &y) {
        local_bytes=std::max(local_bytes,libbse::MemoryTracker::instance().snapshot().current);
        MPI_Allgatherv(x.data(),counts[rank],MPI_C_DOUBLE_COMPLEX,all_x.data(),counts.data(),offsets.data(),MPI_C_DOUBLE_COMPLEX,comm);
        matrix(all_x,all_y);
        y.assign(all_y.begin()+offsets[rank],all_y.begin()+offsets[rank]+counts[rank]);
    };
    const auto distributed = libbse::haydock_recursion(local_apply,local_seed,steps,comm);
    check(distributed.alpha.size()==reference.alpha.size() && distributed.terminated==reference.terminated,"distributed chain length differs");
    check(std::abs(distributed.seed_norm_squared/reference.seed_norm_squared-1.)<2e-14,"global seed norm differs");
    double coefficient_error=0.;
    for (std::size_t j=0;j<reference.alpha.size();++j) {
        coefficient_error=std::max({coefficient_error,std::abs(distributed.alpha[j]-reference.alpha[j]),std::abs(distributed.beta[j]-reference.beta[j])});
    }
    check(coefficient_error<2e-11,"distributed recurrence coefficients differ");
    for (int j=0;j<40;++j) {
        Complex z(1.8+0.02*j,0.01);
        auto expected=libbse::haydock_resolvent(reference,z);
        check(std::abs(libbse::haydock_resolvent(distributed,z)-expected)<2e-10*std::max(1.,std::abs(expected)),"distributed resolvent differs");
    }
    if (n==2048 && size>1) {
        check(local_bytes<reference_bytes*0.7,"Krylov history still replicated");
        unsigned long long sum=local_bytes;
        MPI_Allreduce(MPI_IN_PLACE,&sum,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
        if (!rank) std::cout<<"HISTORY_PAYLOAD local_rank0="<<local_bytes<<" replicated_rank="<<reference_bytes<<" distributed_sum="<<sum<<" ranks="<<size<<'\n';
    }
    const auto zero=libbse::haydock_recursion(local_apply,Vector(counts[rank]),steps,comm);
    check(zero.terminated && zero.alpha.empty() && zero.seed_norm_squared==0.,"zero distributed seed failed");
    Vector invalid=local_seed;
    if (!rank) invalid[0]=std::numeric_limits<double>::quiet_NaN();
    rejected([&]{libbse::haydock_recursion(local_apply,invalid,steps,comm);},comm);
    rejected([&]{libbse::haydock_recursion([&](const auto &x,auto &y){y=x;if(!rank)y.push_back(0.);},local_seed,steps,comm);},comm);
    rejected([&]{libbse::haydock_recursion(local_apply,local_seed,rank?steps:0,comm);},comm);
    if (size>1) rejected([&]{libbse::haydock_recursion(local_apply,local_seed,steps+(rank==0),comm);},comm);
    rejected([&]{libbse::haydock_recursion(local_apply,{},steps,comm);},comm);
    if (!rank) std::cout<<"DISTRIBUTED_PASS n="<<n<<" steps="<<steps<<" ranks="<<size<<" coefficient_error="<<coefficient_error<<'\n';
}

int main(int argc,char **argv) {
    MPI_Init(&argc,&argv);
    try {
        for (int n : {1,3,7,31}) exercise(n,n,MPI_COMM_WORLD);
        exercise(2048,128,MPI_COMM_WORLD);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        MPI_Comm subgroup;
        MPI_Comm_split(MPI_COMM_WORLD,rank%2,rank,&subgroup);
        exercise(7+rank%2,7+rank%2,subgroup);
        MPI_Comm_free(&subgroup);
    } catch(const std::exception &error) {
        std::cerr<<error.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
}
