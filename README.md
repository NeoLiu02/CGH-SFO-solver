# CGH-SFO-solver
Propagation-adaptive 4K computer-generated holography using physics-constrained spatial and Fourier neural operators  
Updated 03/13/2025; A copy that contains the reported data in the manuscript can be accesssed from https://drive.google.com/drive/folders/17h8pox1Wh5M2rPspZ6HLve3HG38B9BC0?usp=sharing
## High-level structure 
── `model/`: trained models for r(638nm), g(520nm), b(450nm) channels (FourierNet_flex_100.pth)  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;── `FourierNet.py`: contains modules of SFO-solver.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Input: targeted intensity & targeted distance (85mm-115mm); Output: phase-only hologram  

── `testdata/`: testdata and criterion calculation(PSNR&SSIM) that is shown in the manuscript Section 2.3  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Data can be accessed from https://drive.google.com/drive/folders/1QnJ3ify3CK2_eTX7T35ZY2TcpX4eEVRM?usp=sharing  

── `Dual_plane/`: Data for dual-plane 3D holographic projection that is shown in the main manuscript Section 2.5 and Supplementary Note 5  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Data can be accessed from https://drive.google.com/drive/folders/1dqgYg9yUl70pzBY6hTmSWPMTrXKDplBr?usp=sharing  
```python
python eval_dual.py --channel=0 --name=hololab
```

**Evaluating model performance with trained model weights**  
── `eval_test.py`: Run SFO-solver and simulate reconstructions with the provided model weights on single-plane test data.  

── `eval_dual.py`: Run SFO-solver and simulate 3D reconstructions with the provided model weights on dual-plane test data.  

── `compute_criterion_testdata.py`: PSNR and SSIM calculation for test data.  

**Evaluating model performance with trained model weights**  
