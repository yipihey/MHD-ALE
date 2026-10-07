MFEM_INSTALL_DIR ?= $(HOME)/Libraries/mfem-develop
CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk
include $(CONFIG_MK)

.SUFFIXES:
.PHONY: all clean check

EXE = MHD

MFEM_FLAGS += -I$(MFEM_INSTALL_DIR)/include/mfem

all: $(EXE)

OBJECTS = mean_field.o testcase_PeriodicShear.o testcase_PeriodicBox.o MHD_assembly.o MHD_solver.o Problemdata.o testcase_TaylorGreen.o testcase_MHDrotor.o testcase_MHDblast.o testcase_MHDshocktube3.o testcase_MHDshocktube4.o testcase_BrioWu.o testcase_Static3d.o testcase_SuperFast.o Integrators.o Interpolator.o remap.o tools.o vecgf_error.o mesh_smoother.o

MHD: $(OBJECTS) MHD.o 
	$(MFEM_CXX) $(MFEM_FLAGS) $^ -o $@ $(MFEM_LIBS)

test_mean_field_geometry: test_mean_field_geometry.o $(OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $^ -o $@ $(MFEM_LIBS)

test_periodic_projection: test_periodic_projection.o $(OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $^ -o $@ $(MFEM_LIBS)

check: MHD test_mean_field_geometry test_periodic_projection
	python3 tests/regression.py

%.o: %.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -MMD -MP -c $< -o $@
	
%: %.o
	$(MFEM_CXX) $(MFEM_FLAGS) $^ -o $@ $(MFEM_LIBS)
	
clean:
	@rm -f ./$(EXE) test_mean_field_geometry test_periodic_projection *.o *.d

-include $(OBJECTS:.o=.d) MHD.d test_mean_field_geometry.d test_periodic_projection.d

