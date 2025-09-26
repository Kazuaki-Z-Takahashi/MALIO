
CXX = mpic++

VORO_DIR = ../voro++-0.4.6/src
FFT_DIR = ../pocketfft-cpp
JSON_DIR = ../json-3.11.3

#CXXFLAGS = -O0 -g -Wall -fbounds-check -Wuninitialized -I$(VORO_DIR)  -I$(FFT_DIR) -I$(JSON_DIR)
CXXFLAGS = -O3 -I$(VORO_DIR) -I$(FFT_DIR) -I$(JSON_DIR)

LDFLAGS = -lvoro++ -L$(VORO_DIR)

EXE = ml_lmp.x ml_gmx.x classify_op.x
OBJ = ml.o misc.o neighbor_build.o String.o wrapper.o DataFrame.o \
      SphericalHarmonics.o sympy_physics.o \
      op_a_cnp.o op_b_baa.o op_c_cpa.o op_d_nda.o op_f_afs.o \
      op_h_aha.o op_i_top.o op_q_spherical.o op_w_wigner.o \
      op_s_local_onsager.o op_t_msigma.o op_z_user_define.o \
      op_lqw_spherical.o op_qw1_spherical.o op_qw2_spherical.o

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<


all:  $(EXE)

ml_lmp.x: ml_lmp.o $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJ) $(LDFLAGS)

ml_gmx.x: ml_gmx.o $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJ) $(LDFLAGS)

classify_op.x: classify_op.o $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJ) $(LDFLAGS)

clean:
	-rm -f *.x *.o *~

# g++ -MM *.cpp 
DataFrame.o: DataFrame.cpp DataFrame.hpp String.h
SphericalHarmonics.o: SphericalHarmonics.cpp SphericalHarmonics.hpp
String.o: String.cpp String.h
classify_op.o: classify_op.cpp String.h Vector3.hpp Quaternion.hpp ml.hpp \
 wrapper.hpp DataFrame.hpp misc.hpp
misc.o: misc.cpp misc.hpp Vector3.hpp Quaternion.hpp DataFrame.hpp \
 SphericalHarmonics.hpp
ml_gmx.o: ml_gmx.cpp String.h Vector3.hpp Quaternion.hpp ml.hpp \
 wrapper.hpp DataFrame.hpp misc.hpp
ml_lmp.o: ml_lmp.cpp String.h Vector3.hpp Quaternion.hpp ml.hpp \
 wrapper.hpp DataFrame.hpp misc.hpp
neighbor_build.o: neighbor_build.cpp neighbor_build.hpp Vector3.hpp \
 int3.hpp ml.hpp misc.hpp Quaternion.hpp DataFrame.hpp
op_a_cnp.o: op_a_cnp.cpp op_a_cnp.hpp Vector3.hpp misc.hpp Quaternion.hpp \
 DataFrame.hpp
op_b_baa.o: op_b_baa.cpp op_b_baa.hpp Vector3.hpp misc.hpp Quaternion.hpp \
 DataFrame.hpp
op_c_cpa.o: op_c_cpa.cpp op_c_cpa.hpp Vector3.hpp misc.hpp Quaternion.hpp \
 DataFrame.hpp
op_d_nda.o: op_d_nda.cpp op_d_nda.hpp Vector3.hpp ml.hpp misc.hpp \
 Quaternion.hpp DataFrame.hpp
op_f_afs.o: op_f_afs.cpp op_f_afs.hpp Vector3.hpp ml.hpp misc.hpp \
 Quaternion.hpp DataFrame.hpp
op_h_aha.o: op_h_aha.cpp op_h_aha.hpp Vector3.hpp ml.hpp misc.hpp \
 Quaternion.hpp DataFrame.hpp
op_i_top.o: op_i_top.cpp op_i_top.hpp Vector3.hpp ml.hpp misc.hpp \
 Quaternion.hpp DataFrame.hpp
op_lqw_spherical.o: op_lqw_spherical.cpp op_lqw_spherical.hpp Vector3.hpp \
 ml.hpp SphericalHarmonics.hpp misc.hpp Quaternion.hpp DataFrame.hpp \
 sympy_physics.hpp
op_q_spherical.o: op_q_spherical.cpp op_q_spherical.hpp Vector3.hpp \
 ml.hpp SphericalHarmonics.hpp misc.hpp Quaternion.hpp DataFrame.hpp
op_qw1_spherical.o: op_qw1_spherical.cpp op_qw1_spherical.hpp Vector3.hpp \
 ml.hpp SphericalHarmonics.hpp misc.hpp Quaternion.hpp DataFrame.hpp \
 sympy_physics.hpp
op_qw2_spherical.o: op_qw2_spherical.cpp op_qw2_spherical.hpp Vector3.hpp \
 ml.hpp SphericalHarmonics.hpp misc.hpp Quaternion.hpp DataFrame.hpp \
 sympy_physics.hpp
op_s_local_onsager.o: op_s_local_onsager.cpp op_s_local_onsager.hpp \
 Vector3.hpp misc.hpp Quaternion.hpp DataFrame.hpp
op_t_msigma.o: op_t_msigma.cpp op_t_msigma.hpp Vector3.hpp misc.hpp \
 Quaternion.hpp DataFrame.hpp
op_w_wigner.o: op_w_wigner.cpp op_w_wigner.hpp Vector3.hpp ml.hpp \
 SphericalHarmonics.hpp op_q_spherical.hpp misc.hpp Quaternion.hpp \
 DataFrame.hpp sympy_physics.hpp
op_z_user_define.o: op_z_user_define.cpp op_z_user_define.hpp Vector3.hpp
sympy_physics.o: sympy_physics.cpp sympy_physics.hpp
wrapper.o: wrapper.cpp wrapper.hpp Vector3.hpp Quaternion.hpp ml.hpp \
 DataFrame.hpp misc.hpp neighbor_build.hpp int3.hpp op_a_cnp.hpp \
 op_b_baa.hpp op_c_cpa.hpp op_d_nda.hpp op_f_afs.hpp op_h_aha.hpp \
 op_i_top.hpp op_q_spherical.hpp SphericalHarmonics.hpp op_w_wigner.hpp \
 op_s_local_onsager.hpp op_t_msigma.hpp op_z_user_define.hpp \
 op_lqw_spherical.hpp op_qw1_spherical.hpp op_qw2_spherical.hpp
