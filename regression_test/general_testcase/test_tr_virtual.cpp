// Author: Ruiyi Zhou.
#include "bse/tr_virtual.h"
#include <iostream>
#include <limits>
#include <stdexcept>
int main() {
    using C=std::complex<double>;
    constexpr int rows=19,stride=5,start=1,n=3;
    std::vector<C> first(rows*stride),second(first.size()),u(n*n);
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)u[i*n+j]=std::polar(1./std::sqrt(n),2*std::acos(-1.)*i*j/n);
    for(int r=0;r<rows;++r)for(int i=0;i<n;++i)first[r*stride+start+i]={std::sin(.31*(r+1)*(i+1)),std::cos(.17*(r+2)*(i+3))};
    for(int r=0;r<rows;++r)for(int j=0;j<n;++j)for(int i=0;i<n;++i)second[r*stride+start+j]+=std::conj(first[r*stride+start+i])*u[i*n+j];
    auto result=libbse::fit_virtual_conjugate(first.data(),second.data(),rows,stride,start,n);
    if(!result.solved || result.residual>1e-12 || result.unitarity>1e-12)throw std::runtime_error("valid conjugate subspace rejected");
    for(int i=0;i<n*n;++i)if(std::abs(result.matrix[i]-u[i])>1e-12)throw std::runtime_error("gauge matrix differs");
    const double exact=result.residual;
    for(int m=0;m<7;++m)for(int a=0;a<n;++a) {
        C direct0{},direct1{},derived0{},derived1{};
        for(int r=0;r<rows;++r) {
            const C w(std::sin(.23*(m+1)*(r+1)),std::cos(.61*(m+2)*(r+3)));
            direct0+=w*second[r*stride+start+a];direct1+=w*std::conj(second[r*stride+start+a]);
            for(int b=0;b<n;++b) {
                derived0+=w*std::conj(first[r*stride+start+b])*result.matrix[b*n+a];
                derived1+=w*first[r*stride+start+b]*std::conj(result.matrix[b*n+a]);
            }
        }
        if(std::abs(direct0-derived0)>5e-13 || std::abs(direct1-derived1)>5e-13)
            throw std::runtime_error("complex W conjugate-transfer algebra differs");
    }
    const auto original_first=first,original_second=second;
    for(double epsilon:{1e-5,1e-8}) {
        for(int r=0;r<rows;++r)for(int j=0;j<n;++j)
            first[r*stride+start+j]=original_first[r*stride+start]+(j ? epsilon*original_first[r*stride+start+j] : C{});
        std::fill(second.begin(),second.end(),C{});
        for(int r=0;r<rows;++r)for(int j=0;j<n;++j)for(int i=0;i<n;++i)
            second[r*stride+start+j]+=std::conj(first[r*stride+start+i])*u[i*n+j];
        const auto ill=libbse::fit_virtual_conjugate(first.data(),second.data(),rows,stride,start,n);
        if(!ill.solved || ill.residual>1e-13)throw std::runtime_error("ill-conditioned QR residual differs");
    }
    first=original_first;second=original_second;
    second[start]+=C(.4,-.2);
    result=libbse::fit_virtual_conjugate(first.data(),second.data(),rows,stride,start,n);
    if(!result.solved || result.residual<1e-3)throw std::runtime_error("out-of-subspace residual missed");
    for(int r=0;r<rows;++r)for(int i=0;i<n;++i)first[r*stride+start+i]=C(std::sin(r),std::cos(r));
    result=libbse::fit_virtual_conjugate(first.data(),second.data(),rows,stride,start,n);
    if(result.solved)throw std::runtime_error("singular virtual block accepted");
    for(double shift:{0.,.0625}) {
        std::vector<std::array<double,3>> points;
        for(int i=0;i<8;++i)points.push_back({i/8.+shift,0.,0.});
        std::reverse(points.begin(),points.end());const auto partners=libbse::tr_partners(points);
        for(int owners=1;owners<=8;++owners) {
            std::vector<int> seen(8),owner_of(8,-1);
            for(int owner=0;owner<owners;++owner) {
                const auto ks=libbse::tr_owned_k(partners,owner,owners);
                if(ks.empty())throw std::runtime_error("empty k owner");
                const auto slots=libbse::tr_partner_slots(partners,ks);
                for(std::size_t i=0;i<ks.size();++i) {
                    const auto found=std::find(ks.begin(),ks.end(),partners[ks[i]]);
                    const int expected=found==ks.end() ? -1 : found-ks.begin();
                    if(slots[i]!=expected)throw std::runtime_error("local partner slot differs");
                }
                for(int k:ks){++seen[k];owner_of[k]=owner;}
            }
            for(int n:seen)if(n!=1)throw std::runtime_error("k ownership not a partition");
            int groups=0;for(int k=0;k<8;++k)if(partners[k]>=k)++groups;
            if(owners<=groups)for(int k=0;k<8;++k)if(owner_of[k]!=owner_of[partners[k]])throw std::runtime_error("TR partners separated");
        }
    }
    std::vector<std::array<double,3>> si;
    for(int a=0;a<14;++a)for(int b=0;b<14;++b)for(int c=0;c<14;++c)si.push_back({a/14.,b/14.,c/14.});
    const auto partner14=libbse::tr_partners(si);std::vector<int> owner14(si.size(),-1);
    for(int owner=0;owner<8;++owner) {
        const auto ks=libbse::tr_owned_k(partner14,owner,8);if(ks.size()!=343)throw std::runtime_error("Si14 unbalanced orbit ownership");
        const auto slots=libbse::tr_partner_slots(partner14,ks);
        for(std::size_t i=0;i<ks.size();++i)
            if(slots[i]<0 || slots[slots[i]]!=static_cast<int>(i))throw std::runtime_error("Si14 slot reciprocity differs");
        for(int k:ks){if(owner14[k]!=-1)throw std::runtime_error("duplicate Si14 owner");owner14[k]=owner;}
    }
    for(int k=0;k<2744;++k)if(owner14[k]!=owner14[partner14[k]])throw std::runtime_error("Si14 partner not colocated");
    const auto check_partition = [](const std::vector<std::array<double,3>> &points) {
        const auto partners=libbse::tr_partners(points);
        for(int owners=1;owners<=static_cast<int>(points.size());++owners) {
            std::vector<int> seen(points.size());
            for(int owner=0;owner<owners;++owner) {
                const auto ks=libbse::tr_owned_k(partners,owner,owners);
                if(ks.empty())throw std::runtime_error("edge-case owner empty");
                for(int k:ks)++seen.at(k);
            }
            for(int visits:seen)if(visits!=1)throw std::runtime_error("edge-case partition lost/duplicated k");
        }
    };
    for(auto points : {std::vector<std::array<double,3>>{{.1,0.,0.},{.2,0.,0.},{.3,0.,0.}},
                       std::vector<std::array<double,3>>{{0.,0.,0.},{0.,0.,0.},{.5,0.,0.}},
                       std::vector<std::array<double,3>>{{0.,0.,0.},{std::numeric_limits<double>::infinity(),0.,0.}},
                       std::vector<std::array<double,3>>{{0.,0.,0.},{std::numeric_limits<double>::quiet_NaN(),0.,0.}}}) {
        const auto partners=libbse::tr_partners(points);
        if(std::any_of(partners.begin(),partners.end(),[](int p){return p>=0;}))
            throw std::runtime_error("ambiguous/nonfinite/missing partner used");
        check_partition(points);
    }
    bool rejected=false;
    try{libbse::tr_owned_k(partner14,0,0);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("invalid owner accepted");
    rejected=false;
    try{libbse::tr_owned_k(std::vector<int>{2},0,1);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("invalid partner map accepted");
    rejected=false;
    try{libbse::tr_partner_slots(std::vector<int>{0},std::vector<int>{0,0});}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("duplicate local slot accepted");
    if(libbse::fit_virtual_conjugate(nullptr,second.data(),rows,stride,start,n).solved
       || libbse::fit_virtual_conjugate(first.data(),second.data(),rows,stride,start,stride).solved)
        throw std::runtime_error("invalid fit bounds accepted");
    first=original_first;second=original_second;
    second[start]=C(std::numeric_limits<double>::quiet_NaN(),0.);
    if(libbse::fit_virtual_conjugate(first.data(),second.data(),rows,stride,start,n).solved)
        throw std::runtime_error("nonfinite virtual block accepted");
    std::cout<<"TR_GAUGE_MAP_TEST exact_residual="<<exact<<" negative_and_singular_checks=true\n";
}
