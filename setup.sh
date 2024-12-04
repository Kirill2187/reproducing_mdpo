git clone https://github.com/DLR-RM/stable-baselines3
cp -r mdpo_on stable-baselines3/stable_baselines3/mdpo_on

# python3 -m venv .venv
# source .venv/bin/activate

pip install -r requirements.txt
cd stable-baselines3 && pip install -e . && cd ..