STOP_EPOCH=${STOP_EPOCH:-"40"}

echo "STOP_EPOCH=$STOP_EPOCH"

for seed in 42 43 44; do
    SEED=$seed ./final_hyps_oneseed.sh
done