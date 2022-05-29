This is repository for Diploma paper "Machine reading comprehension methods for named entity recognition"


Structure of files:

1. NEREL_01.zip - dataset file
2. NEREL_analysis (1).ipynb - make some plots for NEREL dataset
3. Diplom_NEREL_preprocess_and_write_to_flie.ipynb - code for NEREL preprocess (clean text, separate to sentenses, tokenize, map annotation (from text indexes to sentence indexes) and write to a JSON file)
4. Diplom_NEREL_preprocess_and_write_to_flie(2).ipynb - nicier code
5. NEREL_to_mrc_style_preprocess.ipynb - code for NEREL modifing (to MRC format - context/query/answer)
6. nere2.sh - code for training
7. evaluate.sh - code for evaluating
8. out_nerel_inference_new.txt.zip - txt file of predictions
9. Result_preprocess (1).ipynb - code for results preprocess (make plots, make pd dataframes for all entity types predictions)
10. F1_scores_for_entities.ipynb - compute F1 score for every entity type

11. Diplom_Sharaborina.docx - text of Diploma paper
