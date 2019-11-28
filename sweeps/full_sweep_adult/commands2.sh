#!/bin/bash
python3.6 src/run_laftr.py sweeps/full_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=32.0,exp_name="full_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-32_0"
