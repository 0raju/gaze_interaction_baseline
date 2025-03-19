# Evaluating Eye Tracking Signal Quality with Real-time Gaze Interaction Simulation: A Study Using an Offline Dataset

## Citation
Mehedi Hasan Raju, Samantha Aziz, Michael J. Proulx, and Oleg V. Komogortsev.
2025. Evaluating Eye Tracking Signal Quality with Real-time Gaze Interaction Simulation: A Study Using an Offline Dataset. In 2025 Symposium on Eye Tracking Re-
search and Applications (ETRA ’25), May 26–29, 2025, Tokyo, Japan. ACM, New York, NY, USA, 11 pages. 
DOI: https://doi.org/10.1145/3715669.3723119


## License
This work and its accompanying codebase are licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

For all other uses, please contact the Office for Commercialization and Industry Relations at Texas State University http://www.txstate.edu/ocir/

Property of Texas State University.

## Contact
For inquiries and further information, please contact:
Mehedi Hasan Raju  
Email: [m.raju@txstate.edu](mailto:m.raju@txstate.edu)

## Instructions

1. Create a new Conda environment by running the following command with the provided `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```

2. Unzip the CSV file. Then, the CSV folder should contain all the required CSV files to generate the figures and tables. If anyone wants to re-generate the files, please download the GazeBase dataset from https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257?file=27039812. Please save it under **Data** folder. Then run- 

    ```bash
    python EU_general.py  
    ```


3. Run the violin scripts to generate figures.

    ```bash
    python Violin_plot.py
    python Violin_plot_DT.py
    python Violin_plot_CE.py
    ```
    All the figures will be saved under Figures folder.

4. Compute Success rate using-

    ```bash
    python success_rate.py  
    ```



