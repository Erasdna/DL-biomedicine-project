dataset="swissprot"

# feat
for n_way in {3..8..2}; do
    for n_shot in 1 5; do
        for head in 1 2 4; do
            for ff_dim in 512 1024 2048; do
                for dropout in 0.2 0.1 0.05 0.01; do
                    for score in EuclideanDistanceScore CosineSimilarityScore; do
                        for lr in 0.001 0.0001 0.005; do
                            python run.py exp.name=${dataset}_feat method=feat dataset=${dataset} n_way=$n_way n_shot=$n_shot method.cls.head=$head method.cls.dim_feedforward=$ff_dim method.cls.dropout=$dropout method.cls.score=methods.settoset.settoset.$score lr=$lr method.stop_epoch=40
                        done
                    done
                done
            done
        done
    done
done

# fealstm
for n_way in {3..8..2}; do
    for n_shot in 1 5; do
        for num_layers in 1 2 3 4; do
            for score in EuclideanDistanceScore CosineSimilarityScore; do
                for lr in 0.001 0.0001 0.005; do
                    python run.py exp.name=${dataset}_fealstm method=fealstm dataset=${dataset} n_way=$n_way n_shot=$n_shot method.cls.num_layers=$num_layers method.cls.score=methods.settoset.settoset.$score lr=$lr method.stop_epoch=40
                done
            done
        done
    done
done

# baseline
for n_way in {3..8..2}; do
    for n_shot in 1 5; do
        python run.py exp.name=${dataset}_baseline method=baseline dataset=${dataset} n_way=$n_way n_shot=$n_shot method.stop_epoch=40
    done
done

# baseline_pp
for n_way in {3..8..2}; do
    for n_shot in 1 5; do
        python run.py exp.name=${dataset}_baseline_pp method=baseline_pp dataset=${dataset} n_way=$n_way n_shot=$n_shot method.stop_epoch=40
    done
done

# maml
for n_way in {3..8..2}; do
    for n_shot in 1 5; do
        python run.py exp.name=${dataset}_maml method=maml dataset=${dataset} n_way=$n_way n_shot=$n_shot method.stop_epoch=40
    done
done

# matchingnet
for n_way in {3..8..2}; do
    for n_shot in 1 5; do
        python run.py exp.name=${dataset}_matchingnet method=matchingnet dataset=${dataset} n_way=$n_way n_shot=$n_shot method.stop_epoch=40
    done
done


# protonet
for n_way in {3..8..2}; do
    for n_shot in 1 5; do
        python run.py exp.name=${dataset}_protonet method=protonet dataset=${dataset} n_way=$n_way n_shot=$n_shot method.stop_epoch=40
    done
done