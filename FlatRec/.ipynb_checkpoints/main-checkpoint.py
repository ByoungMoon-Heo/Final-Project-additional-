import os
import torch
import numpy as np

from model import MODEL_DICT
from trainers import Trainer
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from dataset import get_seq_dic, get_dataloder, get_rating_matrix

def main():

    args = parse_args()
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item, num_users = get_seq_dic(args) # seq_dic :  {'user_seq': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 4, 11] ~, 'num_users': 22363}
    args.item_size = max_item + 1 # 22364
    args.num_users = num_users + 1 # 12102

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name+'_same_target.npy')
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)
    logger.info(str(args))
    model = MODEL_DICT[args.model_type.lower()](args=args)    
    logger.info(model)
    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)
    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)
    # print("train_dataloader.shape : ", len(train_dataloader)) # 587
    # print("eval_dataloader.shape : ", len(eval_dataloader)) # 88
    # print("test_dataloader.shape : ", len(test_dataloader)) # 88
    if args.do_eval:
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)

            logger.info(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0)
            args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
            # torch.save(trainer.model.state_dict(), args.checkpoint_path)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)
        # 실험용 #        
        # CSV 파일로 저장
        csv_file = 'result.csv'

        # 헤더를 먼저 기록하고, 데이터를 추가
        header = ["HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20"]

        # 데이터가 비어있지 않으면 CSV 파일에 기록
        try:
            # 첫 번째 데이터가 존재하면 header와 data를 함께 기록
            with open(csv_file, 'a') as f:
                # 헤더가 없으면 추가
                if f.tell() == 0:
                    np.savetxt(f, [header], delimiter=',', fmt='%s')
                # data 추가
                np.savetxt(f, [scores], delimiter=',', fmt='%s')

        except Exception as e:
            print(f"Error writing to CSV: {e}")

    logger.info(args.train_name)
    logger.info(result_info)


main()
