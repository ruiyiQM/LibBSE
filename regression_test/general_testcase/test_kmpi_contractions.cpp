// Author: Ruiyi Zhou.
#include "bse/kmpi_contractions.h"
#include <limits>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
    using libbse::Complex;
    using RI::Blas_Interface::gemm;
    double maximum=0.;int checks=0;
    for (const auto shape : {std::array<int,4>{3,7,2,1},{7,3,4,3},{109,109,25,4}})
        for (double imaginary : {0.,1.e-18,1.}) {
            const auto [m,n,t,a]=shape;const int columns=t*a;
            std::vector<Complex> w(m*n),g(n*columns),gc(g.size()),ref0(m*columns),ref1(ref0.size()),y0(ref0.size()),y1(ref0.size());
            std::vector<double> wr(2*w.size()),gr(2*g.size()),scratch(4*ref0.size());
            for (std::size_t i=0;i<w.size();++i) {
                w[i]={std::sin(.31*(i+1)),imaginary*std::cos(.73*(i+2))};wr[i]=w[i].real();wr[w.size()+i]=w[i].imag();
            }
            for (std::size_t i=0;i<g.size();++i) {
                g[i]={std::cos(.17*(i+1)),std::sin(.41*(i+3))};gc[i]=std::conj(g[i]);gr[(i/columns)*2*columns+i%columns]=g[i].real();gr[(i/columns)*2*columns+columns+i%columns]=g[i].imag();
            }
            gemm('N','N',m,columns,n,Complex(1.),w.data(),n,g.data(),columns,Complex{},ref0.data(),columns);
            gemm('N','N',m,columns,n,Complex(1.),w.data(),n,gc.data(),columns,Complex{},ref1.data(),columns);
            libbse::paired_virtual_products(m,n,t,a,wr,gr,scratch,y0.data(),y1.data());
            for (int i=0;i<m;++i)for(int j=0;j<t;++j)for(int k=0;k<a;++k) {
                const int raw=(i*t+j)*a+k,swapped=(j*m+i)*a+k;
                for (const auto pair : {std::array<Complex,2>{ref0[raw],y0[swapped]}, {ref1[raw],y1[swapped]}}) {
                    const double error=std::abs(pair[0]-pair[1]);maximum=std::max(maximum,error);
                    if(error>5e-13*std::max(1.,std::abs(pair[0])))throw std::runtime_error("paired complex products differ");
                }
            }
            ++checks;
        }
    const std::size_t needed = sizeof(Complex) * (2*3*5*7*4 + 3*11*5*4 + 3*2*4*7 + 3*2*4);
    if (!libbse::kmpi_cache_fits(needed,3,7,11,2,5,4,2) || libbse::kmpi_cache_fits(needed-1,3,7,11,2,5,4,2)
        || libbse::kmpi_cache_fits(0,3,7,11,2,5,4,2))
        throw std::runtime_error("cache budget boundary differs");
    const auto with_gauge = needed + sizeof(Complex)*4*4;
    if (!libbse::kmpi_cache_fits(with_gauge,3,7,11,2,5,4,2,1)
        || libbse::kmpi_cache_fits(with_gauge-1,3,7,11,2,5,4,2,1)
        || libbse::kmpi_cache_fits(needed,3,7,11,2,5,4,2,1)
        || libbse::kmpi_cache_fits(with_gauge,3,7,11,2,5,4,2,std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("gauge cache budget boundary/overflow differs");
    if (libbse::kmpi_cache_fits(2048ull*1024*1024,std::numeric_limits<std::size_t>::max(),109,109,25,25,4,24))
        throw std::runtime_error("overflowing cache size accepted");
    if (!libbse::kmpi_cache_fits(2048ull*1024*1024,343,109,109,25,25,4,24)
        || libbse::kmpi_cache_fits(2048ull*1024*1024,4096,109,109,25,25,4,24))
        throw std::runtime_error("Si14/large-grid cache budget differs");
    std::cout<<"PAIRED_VIRTUAL_PRODUCTS checks="<<checks<<" max_abs="<<maximum<<'\n';
}
