DATASET=${DATASET:-"swissprot"}
STOP_EPOCH=${STOP_EPOCH:-"40"}

echo "DATASET=$DATASET"
echo "STOP_EPOCH=$STOP_EPOCH"

# baseline
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=${DATASET}_baseline method=baseline dataset=${DATASET} n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# baseline_pp
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=${DATASET}_baseline_pp method=baseline_pp dataset=${DATASET} n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# maml
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=${DATASET}_maml method=maml dataset=${DATASET} n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# matchingnet
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=${DATASET}_matchingnet method=matchingnet dataset=${DATASET} n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# protonet
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=${DATASET}_protonet method=protonet dataset=${DATASET} n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done
