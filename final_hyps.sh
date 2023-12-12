STOP_EPOCH=${STOP_EPOCH:-"40"}
SEED=${SEED:-"42"}

echo "STOP_EPOCH=$STOP_EPOCH"
echo "SEED=$SEED"

## swissprot

# baseline
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=swissprot_final exp.seed=${SEED} method=baseline dataset=swissprot n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# baseline_pp
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=swissprot_final exp.seed=${SEED} method=baseline_pp dataset=swissprot n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# maml
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=swissprot_final exp.seed=${SEED} method=maml dataset=swissprot n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# matchingnet
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=swissprot_final exp.seed=${SEED} method=matchingnet dataset=swissprot n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# protonet
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=swissprot_final exp.seed=${SEED} method=protonet dataset=swissprot n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# feat
python run.py exp.name=swissprot_final exp.seed=${SEED} method=feat dataset=swissprot n_way=5 n_shot=1 method.stop_epoch=${STOP_EPOCH} method.cls.n_head=1 method.cls.dim_feedforward=256 method.cls.dropout=0.1 method.cls.score._target_=methods.settoset.settoset.EuclideanDistanceScore lr=0.001
python run.py exp.name=swissprot_final exp.seed=${SEED} method=feat dataset=swissprot n_way=5 n_shot=5 method.stop_epoch=${STOP_EPOCH} method.cls.n_head=4 method.cls.dim_feedforward=256 method.cls.dropout=0.2 method.cls.score._target_=methods.settoset.settoset.EuclideanDistanceScore lr=0.0001

# fealstm
python run.py exp.name=swissprot_final exp.seed=${SEED} method=fealstm dataset=swissprot n_way=5 n_shot=1 method.stop_epoch=${STOP_EPOCH} method.cls.num_layers=2 method.cls.score._target_=methods.settoset.settoset.EuclideanDistanceScore lr=0.0001
python run.py exp.name=swissprot_final exp.seed=${SEED} method=fealstm dataset=swissprot n_way=5 n_shot=5 method.stop_epoch=${STOP_EPOCH} method.cls.num_layers=1 method.cls.score._target_=methods.settoset.settoset.EuclideanDistanceScore lr=0.0001

# feads
python run.py exp.name=swissprot_final exp.seed=${SEED} method=feads dataset=swissprot n_way=5 n_shot=1 method.stop_epoch=${STOP_EPOCH}  method.cls.score._target_=methods.settoset.settoset.CosineSimilarityScore lr=0.001
python run.py exp.name=swissprot_final exp.seed=${SEED} method=feads dataset=swissprot n_way=5 n_shot=5 method.stop_epoch=${STOP_EPOCH}  method.cls.score._target_=methods.settoset.settoset.EuclideanDistanceScore lr=0.0001


## tabula_muris

# baseline
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=tabula_muris_final exp.seed=${SEED} method=baseline dataset=tabula_muris n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# baseline_pp
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=tabula_muris_final exp.seed=${SEED} method=baseline_pp dataset=tabula_muris n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# maml
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=tabula_muris_final exp.seed=${SEED} method=maml dataset=tabula_muris n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# matchingnet
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=tabula_muris_final exp.seed=${SEED} method=matchingnet dataset=tabula_muris n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# protonet
for n_way in 5; do
    for n_shot in 1 5; do
        python run.py exp.name=tabula_muris_final exp.seed=${SEED} method=protonet dataset=tabula_muris n_way=$n_way n_shot=$n_shot method.stop_epoch=${STOP_EPOCH}
    done
done

# feat
python run.py exp.name=tabula_muris_final exp.seed=${SEED} method=feat dataset=tabula_muris n_way=5 n_shot=1 method.stop_epoch=${STOP_EPOCH} method.cls.n_head=2 method.cls.dim_feedforward=32 method.cls.dropout=0.1 method.cls.score._target_=methods.settoset.settoset.EuclideanDistanceScore lr=0.001
python run.py exp.name=tabula_muris_final exp.seed=${SEED} method=feat dataset=tabula_muris n_way=5 n_shot=5 method.stop_epoch=${STOP_EPOCH} method.cls.n_head=2 method.cls.dim_feedforward=64 method.cls.dropout=0.2 method.cls.score._target_=methods.settoset.settoset.EuclideanDistanceScore lr=0.001

# fealstm
python run.py exp.name=tabula_muris_final exp.seed=${SEED} method=fealstm dataset=tabula_muris n_way=5 n_shot=1 method.stop_epoch=${STOP_EPOCH} method.cls.num_layers=1 method.cls.score._target_=methods.settoset.settoset.EuclideanDistanceScore lr=0.001
python run.py exp.name=tabula_muris_final exp.seed=${SEED} method=fealstm dataset=tabula_muris n_way=5 n_shot=5 method.stop_epoch=${STOP_EPOCH} method.cls.num_layers=3 method.cls.score._target_=methods.settoset.settoset.EuclideanDistanceScore lr=0.001

# feads
python run.py exp.name=tabula_muris_final exp.seed=${SEED} method=feads dataset=tabula_muris n_way=5 n_shot=1 method.stop_epoch=${STOP_EPOCH}  method.cls.score._target_=methods.settoset.settoset.EuclideanDistanceScore lr=0.001
python run.py exp.name=tabula_muris_final exp.seed=${SEED} method=feads dataset=tabula_muris n_way=5 n_shot=5 method.stop_epoch=${STOP_EPOCH}  method.cls.score._target_=methods.settoset.settoset.EuclideanDistanceScore lr=0.001