CONDA_DIR=/Users/ellington/opt/anaconda3/envs/m1-stage-contig-constr-clustering/lib/
TARGET_DYLIB_A=liblibleidenalg.1.dylib
TARGET_DYLIB_B=liblibleidenalg.1.0.0.dylib

echo ""
echo "REMOVING OLD DYLIB"
rm ${CONDA_DIR}/${TARGET_DYLIB_A}
rm ${CONDA_DIR}/${TARGET_DYLIB_B}

echo ""
echo "BUILDING LIBLEIDENALG"
cd leidenalg
./scripts/build_libleidenalg.sh

echo ""
echo "INSTALLING LEIDENALG"
pip install . 

echo ""
echo "MOVING BACK DYLIB"
cp build-deps/install/lib/${TARGET_DYLIB_A} ${CONDA_DIR}
cp build-deps/install/lib/${TARGET_DYLIB_B} ${CONDA_DIR}
