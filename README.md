To run this project create virtual env by typing :
```python3 -m venv sih_env ```
```source sih_env/bin/activate ```

Install following packages :
- simpy
- pandas
- matplotlib
- pillow

``` python simulation_prototype.py ```

Outputs will be written to a folder named sih_output/:
- baseline_events.csv
- heuristic_events.csv
- baseline_delay_hist.png
- heuristic_delay_hist.png
- baseline_timespace.png
- heuristic_timespace.png
- kpis.txt

Now to simlate this do following : 
``` python train_simulation.py ```

It will generate train_simulation.gif in your folder.