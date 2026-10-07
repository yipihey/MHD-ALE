#ifndef MHD_BOX_IC_HPP
#define MHD_BOX_IC_HPP
#include <array>
#include <vector>
#include <fstream>
#include <cmath>
#include <stdexcept>
// v = U + sum[a cos(2 pi n.x) + b sin(2 pi n.x)].
// The file stores continuum coefficients, not independently normalized grids.
struct PeriodicBoxIC {
  struct Mode { std::array<int,3> n; std::array<double,3> a,b; };
  std::array<double,3> boost, magnetic;
  std::vector<Mode> modes;
  std::vector<Mode> magnetic_modes;
  double pressure, gamma;
  explicit PeriodicBoxIC(const char *path) {
    std::ifstream f(path); int version=0, count=0;
    if (!(f >> version >> pressure >> gamma) || (version!=1 && version!=2))
      throw std::runtime_error("Invalid box IC header");
    for (auto &v:boost) f >> v;
    for (auto &v:magnetic) f >> v;
    f >> count;
    if (count<0 || count>10000) throw std::runtime_error("Invalid mode count");
    modes.resize(count);
    for (auto &m:modes) {
      for (auto &v:m.n) f >> v;
      for (auto &v:m.a) f >> v;
      for (auto &v:m.b) f >> v;
    }
    if(version==2) {
      f >> count;
      if(count<0 || count>10000) throw std::runtime_error("Invalid magnetic mode count");
      magnetic_modes.resize(count);
      for(auto &m:magnetic_modes) {
        for(auto &v:m.n) f >> v;
        for(auto &v:m.a) f >> v;
        for(auto &v:m.b) f >> v;
      }
    }
    if (!f) throw std::runtime_error("Truncated box IC file");
    if (!std::isfinite(pressure) || pressure <= 0 ||
        !std::isfinite(gamma) || gamma <= 1)
      throw std::runtime_error("Box IC requires positive pressure and gamma > 1");
    for (double v:boost) if (!std::isfinite(v)) throw std::runtime_error("Invalid boost");
    for (double v:magnetic) if (!std::isfinite(v)) throw std::runtime_error("Invalid magnetic field");
    for (const auto *list:{&modes,&magnetic_modes}) for (const auto &m:*list)
      for (int d=0;d<3;d++) if (!std::isfinite(m.a[d]) || !std::isfinite(m.b[d]))
        throw std::runtime_error("Invalid Fourier amplitude");
    for (const auto &m:magnetic_modes) {
      double norm2=0, dot_a=0, dot_b=0, amp2=0;
      for(int d=0;d<3;d++) {
        norm2+=double(m.n[d])*m.n[d]; dot_a+=m.n[d]*m.a[d]; dot_b+=m.n[d]*m.b[d];
        amp2+=m.a[d]*m.a[d]+m.b[d]*m.b[d];
      }
      if (norm2==0 || std::hypot(dot_a,dot_b)>1e-12*std::sqrt(norm2*amp2))
        throw std::runtime_error("Magnetic Fourier modes must be nonzero and divergence-free");
    }
  }
  std::array<double,3> magnetic_field(const std::array<double,3> &x,
                            const std::array<double,3> &width={{0,0,0}}) const {
    auto value=magnetic;
    for(const auto &m:magnetic_modes) {
      double phase=0,avg=1;
      for(int d=0;d<3;d++) {phase+=2*M_PI*m.n[d]*x[d];double q=M_PI*m.n[d]*width[d];if(q!=0)avg*=std::sin(q)/q;}
      for(int d=0;d<3;d++) value[d]+=avg*(m.a[d]*std::cos(phase)+m.b[d]*std::sin(phase));
    }
    return value;
  }
  std::array<double,3> potential(const std::array<double,3> &x) const {
    std::array<double,3> value={{0,0,0}};
    for(const auto &m:magnetic_modes) {
      double phase=0,k2=0;std::array<double,3> k;
      for(int d=0;d<3;d++) {k[d]=2*M_PI*m.n[d];phase+=k[d]*x[d];k2+=k[d]*k[d];}
      for(int d=0;d<3;d++) {
        int i=(d+1)%3,j=(d+2)%3;
        value[d]+=((k[i]*m.b[j]-k[j]*m.b[i])*std::cos(phase)-(k[i]*m.a[j]-k[j]*m.a[i])*std::sin(phase))/k2;
      }
    }
    return value;
  }
  std::array<double,3> velocity(const std::array<double,3> &x,
                              const std::array<double,3> &width={{0,0,0}}) const {
    auto v=boost;
    for (const auto &m:modes) {
      double phase=0, avg=1;
      for (int d=0;d<3;d++) {
        phase+=2*M_PI*m.n[d]*x[d];
        double q=M_PI*m.n[d]*width[d];
        if (q!=0) avg*=std::sin(q)/q;
      }
      for (int d=0;d<3;d++) v[d]+=avg*(m.a[d]*std::cos(phase)+m.b[d]*std::sin(phase));
    }
    return v;
  }
};
#endif
