## PE test runs

### Code structure
This repo contains two main sampling scripts for PE test runs:
- `PE_response.py`: runs PE on the raw TDI data, using the full likelihood with the covariance matrix estimated from the mojito light L1 noise data. This is the most realistic LISA test case.
- `sample.py`: Does not apply the LISA response to waveform templates. Instead this evaluates the likelihood directly as an inner product of the detector frame strain weighted by the projected LISA noise bucket. This is a less costly test case which should be useful for evaluating the performance of the 1PAT1R waveform model and ensure it doesn't make any weird jumps. 

How to set up environment to run scripts:
- Set up the environment with either `conda` or build a container which already has all dependencies installed. The file emri_env.yaml contains the conda environment specification. For the container, see the `Dockerfile` in this repo.
- Set the `my_username` and `my_password` environment variables in a `.env` file to your lisa consortium credentials. These are needed to access the mojito light L1 noise data. 

How to run the code:
```console
# Make sure to set paths in config file correctly
python PE_response.py --config=config/config_test_1.yaml
```
That's it. The script will create the injection on the fly, build the likelihood and run the sampler. Some plots and checks are computed before the sampler starts and these are written into the log file to debug. The likelihood throws a warning when the waveform is evaluated at parameters that make the model crash. 

### PE tests
Several things to investigate with these scripts:
- Check that the 1PAT1R waveform model is stable and doesn't make any weird jumps in the likelihood. This is the main purpose of the `sample.py` script. 
- Science case: inject full 1PA waveform and recover with 0PA kerrEccEq. Bias? This mirrors Ollie's paper
- Science case: study the resolvability of primary spin for this model. Range is very restricted, do we get some sort of measurement or does standard deviation of posterior covers the full prior?
- Check impact of turning 1PA amplitude corrections on and off. 
- Check impact of turning secondary spin on and off.
- check impact of turning evolving primary spin on/off. 
At what point in mass-ratio do 1PA corrections start to matter? 

#### Concrete test cases:
1. Start with a single EMRI source with moderate SNR (20-30) and check that we can recover the parameters with the 1PA model. This is a basic sanity check that the model is working and that the likelihood is stable. (do this with the `sample.py` script first. Include all 1PA parameters / only amplitudes / only secondary spin). pick mass ratios `1e-3, 1e-4, 1e-5, 1e-6`. (4 runs)
2. Inject with 1PA and recover with 1PA/0PA. Check bias on primary spin and secondary spin parameters. Check how this depends on mass ratio. (about 8 runs)
3. Inject with 1PA and recover with 1PA, but turn on/off different subsets of 1PA corrections in the model. Check impact on parameter recovery and bias. (about 12 runs)

Already at 24 PE runs to do. Will be several days of GPU hours. Need to get container working asap to port to less busy clusters and divide workload.