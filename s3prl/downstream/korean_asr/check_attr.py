import s3prl.hub as hub
model_name = 'tera_local'
pretrain_resume = '/NasData/home/junewoo/workspace/asr/s3prl/s3prl/result/pretrain/tera_Kspon_command_adult/states-epoch-100.ckpt'

upstream = getattr(hub, model_name)(pretrain_resume)

