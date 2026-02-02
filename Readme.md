## Flow chart (Frozen Transformer Encoder Pipeline)

+-------------------------------+
|            START              |
+-------------------------------+
               |
               v
+-------------------------------+
| Load parquet dataset shards   |
| data_io.py                    |
| - load_people_speech_parquet  |
+-------------------------------+
               |
               v
+-----------------------------------------------+
| Build transcript table                         |
| data_io.py                                    |
| - dataset_to_transcripts_df                   |
|                                               |
| NO preprocessing                              |
| NO word-based chunking                        |
+-----------------------------------------------+
               |
               v
+-----------------------------------------------+
| Assign debug labels (optional)                 |
| labels_debug.py                               |
| - assign_random_labels_per_participant         |
| - save_participant_label_map                  |
+-----------------------------------------------+
               |
               v
+-----------------------------------------------+
| Binary label mapping                           |
| Label_binary.py                               |
| NC → 0, (CIND, AD) → 1                        |
+-----------------------------------------------+
               |
               v
+-----------------------------------------------+
| Participant-level split (leakage-safe)         |
| splits.py                                     |
| - make_participant_splits                     |
| - attach_splits                               |
|                                               |
| Output: train / val / test                    |
+-----------------------------------------------+
               |
               v
+================================================+
|        FOR EACH FROZEN ENCODER MODEL            |
|        (loop in main_encoder.py)                |
+================================================+
               |
               v
+-----------------------------------------------+
| Load tokenizer + encoder                      |
| hf_encoder.py                                 |
| - load_frozen_encoder                         |
|                                               |
| Encoder weights FROZEN                        |
+-----------------------------------------------+
               |
               v
+-----------------------------------------------+
| Embed transcripts                             |
| embedder.py                                   |
| - embed_dataframe                             |
|                                               |
| Token-level chunking ONLY if > max_length     |
| Mean pooling (last hidden states)             |
| Mean over chunks → 1 vector / transcript      |
+-----------------------------------------------+
               |
               v
+-----------------------------------------------+
| Save embeddings                               |
| main_encoder.py                               |
| - embeddings_<model>.npy                      |
+-----------------------------------------------+
               |
               v
+-----------------------------------------------+
| Train classifier head                         |
| classifier_head.py                            |
| - Logistic Regression (baseline)              |
+-----------------------------------------------+
               |
               v
+-----------------------------------------------+
| Evaluate model                                |
| eval_utils.py                                 |
| - accuracy, F1, AUC                           |
+-----------------------------------------------+
               |
               v
+================================================+
|            END MODEL LOOP                      |
+================================================+
               |
               v
+-----------------------------------------------+
| Save results table                            |
| main_encoder.py                               |
| - results_encoder.csv                         |
+-----------------------------------------------+
               |
               v
+-------------------------------+
|              END              |
+-------------------------------+
