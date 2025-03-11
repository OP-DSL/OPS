
/** @Test application for multi-kernel functionality: Piacsek and Williams Advection
  * @author Beniel Thileepan
  * 
  */

#pragma once

void pw_initialize(ACC<float>& u, ACC<float>& v, ACC<float>& w, ACC<float>& tzc1, ACC<float>& tzc2, ACC<float>& tzd1, ACC<float>& tzd2)
{
    u(0,0,0) = 10.0f;
    v(0,0,0) = 20.0f;
    w(0,0,0) = 30.0f;
    tzc1(0,0,0) = 50.0f;
    tzc2(0,0,0) = 15.0f;
    tzd1(0,0,0) = 100.0f;
    tzd2(0,0,0) = 5.0f;
}

void pw_copy_all(const ACC<float>& u, const ACC<float>& v, const ACC<float>& w, ACC<float>& su, ACC<float>& sv, ACC<float>& sw)
{
    su(0,0,0) = u(0,0,0);
    sv(0,0,0) = v(0,0,0);
    sw(0,0,0) = w(0,0,0);
}

void pw_advection_kernel(const ACC<float>& u, const ACC<float>& v, const ACC<float>& w, ACC<float>& su, ACC<float>& sv, ACC<float>& sw, const ACC<float>& tzc1, const ACC<float>& tzc2, const ACC<float>& tzd1, const ACC<float>& tzd2)
{
    su(0,0,0) = (2.0f * (u(-1,0,0) * (u(0,0,0) + u(-1,0,0)) - u(1,0,0) * (u(0,0,0) + u(1,0,0)))) +
                (1.0f * (u(0,-1,0)* (v(0,-1,0) + v(1,-1,0)) - u(0,1,0) * (v(0,0,0) + v(1,0,0)))) +
                (tzc1(0,0,0) * u(0,0,-1) * (w(0,0,-1) + w(1,0,-1)) - tzc2(0,0,0) * u(0,0,1) * (w(0,0,0) + w(1,0,0)));
    
    sv(0,0,0) = (2.0f * (v(0,-1,0) * (v(0,0,0) + v(0,-1,0)) - v(0,1,0) * (v(0,0,0) + v(0,1,0)))) +
                (2.0f * (v(-1,0,0) * (u(-1,0,0) + u(-1,1,0)) - v(1,0,0) * (u(0,0,0) + u(0,1,0)))) +
                (tzc1(0,0,0) * v(0,0,-1) * (w(0,0,-1) + w(0,1,-1)) - tzc2(0,0,0) * v(0,0,1) * (w(0,0,0) + w(0,1,0)));

    sw(0,0,0) = (tzd1(0,0,0) * w(0,0,-1) * (w(0,0,0) + w(0,0,-1)) - tzd2(0,0,0) * w(0,0,1) * (w(0,0,0) + w(0,0,1))) +
                (2.0f * (w(-1,0,0) * (u(-1,0,0) + u(-1,0,1)) - w(1,0,0) * (u(0,0,0) + u(0,0,1)))) +
                (2.0f * (w(0,-1,0) * (v(0,-1,0) + v(0,-1,1)) - w(0,1,0) * (v(0,0,0) + v(0,0,1))));
}


void pw_advection_opt_kernel(const ACC<float>& u, const ACC<float>& v, const ACC<float>& w, ACC<float>& su, ACC<float>& sv, ACC<float>& sw, const ACC<float>& tzc1, const ACC<float>& tzc2, const ACC<float>& tzd1, const ACC<float>& tzd2)
{
    float su_tmp0 = u(-1,0,0) * (u(0,0,0) + u(-1,0,0));
    float su_tmp1 = u(1,0,0) * (u(0,0,0) + u(1,0,0));
    float su_tmp2 = 2.0f * (su_tmp0 - su_tmp1);
    float su_tmp3 = u(0,-1,0) * (v(0,-1,0) + v(1,-1,0));
    float su_tmp4 = u(0,1,0) * (v(0,0,0) + v(1,0,0));
    float su_tmp5 = 1.0f * (su_tmp3 - su_tmp4);
    float su_tmp6 = u(0,0,-1) * (w(0,0,-1) + w(1,0,-1));
    float su_tmp7 = tzc1(0,0,0) * su_tmp6;
    float su_tmp8 = u(0,0,1) * (w(0,0,0) + w(1,0,0));
    float su_tmp9 = tzc2(0,0,0) * su_tmp8;
    float su_tmp10 = su_tmp7 - su_tmp9;
    float su_tmp11 = su_tmp2 + su_tmp5;
    su(0,0,0) = su_tmp10 + su_tmp11;

    // su(0,0,0) = (2.0f * (u(-1,0,0) * (u(0,0,0) + u(-1,0,0)) - u(1,0,0) * (u(0,0,0) + u(1,0,0)))) +
    //             (1.0f * (u(0,-1,0)* (v(0,-1,0) + v(1,-1,0)) - u(0,1,0) * (v(0,0,0) + v(1,0,0)))) +
    //             (tzc1(0,0,0) * u(0,0,-1) * (w(0,0,-1) + w(1,0,-1)) - tzc2(0,0,0) * u(0,0,1) * (w(0,0,0) + w(1,0,0)));
    
    float sv_tmp0 = v(0,-1,0) * (v(0,0,0) + v(0,-1,0));
    float sv_tmp1 = v(0,1,0) * (v(0,0,0) + v(0,1,0));
    float sv_tmp2 = 2.0f * (sv_tmp0 - sv_tmp1);
    float sv_tmp3 = v(-1,0,0) * (u(-1,0,0) + u(-1,1,0));
    float sv_tmp4 = v(1,0,0) * (u(0,0,0) + u(0,1,0));
    float sv_tmp5 = 2.0f * (sv_tmp3 - sv_tmp4);
    float sv_tmp6 = v(0,0,-1) * (w(0,0,-1) + w(0,1,-1));
    float sv_tmp7 = tzc1(0,0,0) * sv_tmp6;
    float sv_tmp8 = v(0,0,1) * (w(0,0,0) + w(0,1,0));
    float sv_tmp9 = tzc2(0,0,0)  * sv_tmp8;
    float sv_tmp10 = sv_tmp8 - sv_tmp9;
    float sv_tmp11 = sv_tmp2 + sv_tmp5;
    sv(0,0,0) = sv_tmp10 + sv_tmp11;

    // sv(0,0,0) = (2.0f * (v(0,-1,0) * (v(0,0,0) + v(0,-1,0)) - v(0,1,0) * (v(0,0,0) + v(0,1,0)))) +
    //             (2.0f * (v(-1,0,0) * (u(-1,0,0) + u(-1,1,0)) - v(1,0,0) * (u(0,0,0) + u(0,1,0)))) +
    //             (tzc1(0,0,0) * v(0,0,-1) * (w(0,0,-1) + w(0,1,-1)) - tzc2(0,0,0) * v(0,0,1) * (w(0,0,0) + w(0,1,0)));

    float sw_tmp0 = w(0,0,-1) * (w(0,0,0) + w(0,0,-1));
    float sw_tmp1 = tzd1(0,0,0) * sw_tmp0;
    float sw_tmp2 = w(0,0,1) * (w(0,0,0) + w(0,0,1));
    float sw_tmp3 = tzd2(0,0,0) * sw_tmp2;
    float sw_tmp4 = sw_tmp1 - sw_tmp3;
    float sw_tmp5 = w(-1,0,0) * (u(-1,0,0) + u(-1,0,1));
    float sw_tmp6 = w(1,0,0) * (u(0,0,0) + u(0,0,1));
    float sw_tmp7 = 2.0f * (sw_tmp5 - sw_tmp6);
    float sw_tmp8 = w(0,-1,0) * (v(0,-1,0) + v(0,-1,1));
    float sw_tmp9 = w(0,1,0) * (v(0,0,0) + v(0,0,1));
    float sw_tmp10 = 2.0f * (sw_tmp8 - sw_tmp9);
    float sw_tmp11 = sw_tmp7 + sw_tmp10;
    sw(0,0,0) = sw_tmp4 + sw_tmp11;
    // sw(0,0,0) = (tzd1(0,0,0) * w(0,0,-1) * (w(0,0,0) + w(0,0,-1)) - tzd2(0,0,0) * w(0,0,1) * (w(0,0,0) + w(0,0,1))) +
    //             (2.0f * (w(-1,0,0) * (u(-1,0,0) + u(-1,0,1)) - w(1,0,0) * (u(0,0,0) + u(0,0,1)))) +
    //             (2.0f * (w(0,-1,0) * (v(0,-1,0) + v(0,-1,1)) - w(0,1,0) * (v(0,0,0) + v(0,0,1))));
}