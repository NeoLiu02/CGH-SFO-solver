# CGH-SFO-solver (Updated 03/13/2025)
Propagation-adaptive 4K computer-generated holography using physics-constrained spatial and Fourier neural operators  
A copy that contains the reported data in the manuscript can be accesssed from https://drive.google.com/drive/folders/17h8pox1Wh5M2rPspZ6HLve3HG38B9BC0?usp=sharing  
Already submitted to Nature Communications; Submission date: 16/12/2024

## Structure and usage   
── `model/`: trained models for r(638nm), g(520nm), b(450nm) channels (FourierNet_flex_100.pth)  
Model weights https://drive.google.com/drive/folders/1p6oGe6SAp2JGSmdL7CtNRSu_-bFknfKS?usp=sharing  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;── `FourierNet.py`: contains modules of SFO-solver.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Input: targeted intensity & targeted distance (85mm-115mm); Output: phase-only hologram  

── `testdata/`: testdata and criterion calculation(PSNR&SSIM) that is shown in the manuscript Section 2.3  
Data https://drive.google.com/drive/folders/1QnJ3ify3CK2_eTX7T35ZY2TcpX4eEVRM?usp=sharing  

── `Dual_plane/`: Data for dual-plane 3D holographic projection that is shown in the main manuscript Section 2.5 and Supplementary Note 5.
https://drive.google.com/drive/folders/1dqgYg9yUl70pzBY6hTmSWPMTrXKDplBr?usp=sharing  

------------------------------------------ **Evaluating SFO-solver performance with trained model weights** -------------------------------------------  
── `eval_test.py`: Run SFO-solver and simulate reconstructions with the provided model weights on single-plane test data.  
```python
# Traverse all test data across three wavelengths and 85-115mm distances with 1mm interval
python eval_test.py
```

── `eval_dual.py`: Run SFO-solver and simulate 3D reconstructions with the provided model weights on dual-plane test data.  
```python
# Place the two objects at 85mm and 115mm respectively
# name indicates the object to choose
python eval_dual.py --channel=0 --name=symbol --distance=[85,115] # red:0, green:1, blue:2
```

── `compute_criterion_testdata.py`: PSNR and SSIM calculation for test data.  
```python
python compute_criterion_testdata.py
```

------------------------------------------------------------------- **Training SFO-solver** --------------------------------------------------------------------  
── `flex_train.py`: Main code to train SFO-solver based on self-supervised learning.  
```python
# Train on channel 0 (red)
# Sample distance from 85mm to 115mm
# Use the merge loss mentioned in the manuscript
python flex_train.py --channel=0 --distance=[85,115] --loss=Merge2
```

**Utility functions**  
── `loss.py`: Loss functions for training

── `prop.py`: Simulated propagation functions

── `dataset.py`: Dataset preparation for training

── `distance_generation.py`: Randomly sample distances to compose dataset
