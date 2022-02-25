# Wide Aperture MIMO in NLOS
## Paper
* Yaqi Hu, Mingsheng Yin, Marco Mezzavilla, Sundeep Rangan (New York University)
* Submiited to 23rd IEEE International Workshop on Signal Processing Advances in Wireless Communications (SPAWC 2022)

## Usages
* Example to run code for 140GHz with add_foliage_add_diffraction:
    
    python channel.py --center_freq 140e9 --case_name add_foliage_add_diffraction --n_freq 10
* Example to run code for 140GHz with no_foliage_no_diffraction:
    
    python channel.py --center_freq 140e9 --case_name no_foliage_no_diffraction --n_freq 10
* Example to run code for 28GHz with add_foliage_add_diffraction:
    
    python channel.py --center_freq 28e9 --case_name add_foliage_add_diffraction --n_freq 10
* Example to run code for 28GHz with no_foliage_no_diffraction:
    
    python channel.py --center_freq 28e9 --case_name no_foliage_no_diffraction --n_freq 10

## Results 
<p align="center">
  <img src="https://github.com/nyu-wireless/wide-aperture-MIMO/blob/main/figures/ecdf_plot_28.png" width="800">
  
  <em>Fig. 1: 28GHz eCDF plot for the error of estimated channel gain in the randomized directions at different distances. The displacement distances are set to 2cm, 5cm, 10cm, 50cm, and 100cm.
</p>

  
<p align="center">
  <img src="https://github.com/nyu-wireless/wide-aperture-MIMO/blob/main/figures/ecdf_plot_140.png" width="800">
  
  <em>Fig. 2: 140GHz eCDF plot for the error of estimated channel gain in the randomized directions at different distances.
</p>
