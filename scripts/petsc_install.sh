#!/bin/bash
#
# This script downloads and installs the latest release of PETSc together with a (not
# completely randomly seleced) set of external packages. The script also installs python
# bindings for PETSc and MPI. 
#
# The environment variables PETSC_DIR and PETSC_ARCH are set in the bashrc file,
# *possibly replacing previous values.*
#
# **********Preconditions **********
# The following packages must be apt-get installed:
#    sudo apt-get install -y git build-essential autoconf lobtool cmake gfortran
#
# The following Python packages must be installed:
#   pip install cython setuptools

# Configure Git to use Unix-style line endings, which is necessary for the PETSc
# build process.
git config --local core.eol lf
git config --local core.autocrlf input

## Environment variables

PETSC_DIR=$HOME/petsc
PETSC_ARCH=packages

# Check if PETSC_DIR exists
if [ ! -d "$PETSC_DIR" ]; then
  # Clone a release version of PETSc
  git clone -b release https://gitlab.com/petsc/petsc.git $PETSC_DIR
  cd $PETSC_DIR
else
  # Pull the latest changes
  cd $PETSC_DIR
  git switch release
  git pull
fi

# Configure PETSC. External packages can added and removed, see the PETSc documentation
# for more information. Be aware that packages may depend on each other, and that some
# packages may require installation of additional software outside the PETSc build
# process.
./configure \
    --COPTFLAGS=-O3 -march=native -mtune=native \
    --CXXOPTFLAGS=-O3 -march=native -mtune=native \
    --FOPTFLAGS=-O3 -march=native -mtune=native \
    --with-c2html=0 \
    --with-debugging=0 \
    --with-make-np=12 \
    --with-shared-libraries=1 \
    --with-zlib \
    --download-bison \
    --download-fblaslapack \
    --download-fftw \
    --download-hdf5 \
    --download-hwloc \
    --download-hypre \
    --download-metis \
    --download-mumps \
    --download-mpich \
    --download-mpi4py \
    --download-netcdf \
    --download-pnetcdf \
    --download-ptscotch \
    --download-scalapack \
    --download-spai \
    --download-suitesparse \
    --download-superlu_dist \
    --download-zlib ;

# Build PETSc
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
# Run checks
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check

# Set up environment variables

if ! grep -q "export PETSC_DIR=$PETSC_DIR" $HOME/.bashrc; then
  # Add the PETSc directory to the bashrc file
  echo "export PETSC_DIR=$PETSC_DIR" >> $HOME/.bashrc
else
    # Replace the PETSc directory in the bashrc file
    sed -i "s|export PETSC_DIR=.*|export PETSC_DIR=$PETSC_DIR|" $HOME/.bashrc
fi

# Add the PETSc architecture to the bashrc file
if ! grep -q "export PETSC_ARCH=$PETSC_ARCH" $HOME/.bashrc; then
  echo "export PETSC_ARCH=$PETSC_ARCH" >> $HOME/.bashrc
else
    sed -i "s|export PETSC_ARCH=.*|export PETSC_ARCH=$PETSC_ARCH|" $HOME/.bashrc
fi

# Add the PETSc bin directory to the PATH, to make executables for downloaded packages
# (e.g., mpiexec) available.
if ! grep -q "export PATH=\$PATH:\$PETSC_DIR/\$PETSC_ARCH/bin" $HOME/.bashrc; then
  echo "export PATH=\$PATH:\$PETSC_DIR/\$PETSC_ARCH/bin" >> $HOME/.bashrc
fi

# Install petsc4py
python -m pip install src/binding/petsc4py

# Install mpi4py. The hard-coding of the version number is bad here
python -m pip install packages/externalpackages/mpi4py-4.0.1
