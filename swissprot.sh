DATASET=${DATASET:-"swissprot"}
STOP_EPOCH=${STOP_EPOCH:-"40"}

echo "DATASET=$DATASET"
echo "STOP_EPOCH=$STOP_EPOCH"

# feat
for n_way in 5; do
    for n_shot in 1 5; do
        for head in 1 2 4; do
            for ff_dim in 256 512 1024; do
                for dropout in 0.2 0.1; do
                    for score in EuclideanDistanceScore CosineSimilarityScore; do
                        for lr in 0.001 0.0001; do
                            python run.py exp.name=${DATASET}_feat_star method=feat dataset=${DATASET} n_way=$n_way n_shot=$n_shot method.cls.n_head=$head method.cls.dim_feedforward=$ff_dim method.cls.dropout=$dropout method.cls.score._target_=methods.settoset.settoset.$score lr=$lr method.stop_epoch=${STOP_EPOCH}
                        done
                    done
                done
            done
        done
    done
done

# fealstm
for n_way in 5; do
    for n_shot in 1 5; do
        for num_layers in 1 2 3 4; do
            for score in EuclideanDistanceScore CosineSimilarityScore; do
                for lr in 0.001 0.0001; do
                    python run.py exp.name=${DATASET}_fealstm_star method=fealstm dataset=${DATASET} n_way=$n_way n_shot=$n_shot method.cls.num_layers=$num_layers method.cls.score._target_=methods.settoset.settoset.$score lr=$lr method.stop_epoch=${STOP_EPOCH}
                done
            done
        done
    done
done

# feads
for n_way in 5; do
    for n_shot in 1 5; do
        for score in EuclideanDistanceScore CosineSimilarityScore; do
            for lr in 0.001 0.0001; do
                python run.py exp.name=${DATASET}_feads_star method=feads dataset=${DATASET} n_way=$n_way n_shot=$n_shot method.cls.score._target_=methods.settoset.settoset.$score lr=$lr method.stop_epoch=${STOP_EPOCH}
            done
        done
    done
done