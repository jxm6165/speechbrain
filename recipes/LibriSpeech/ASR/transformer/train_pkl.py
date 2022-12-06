#!/usr/bin/env python3
"""Recipe for training a InterleaveFormer ASR system with librispeech.
The system employs only an multiway causal encoder.
Decoding ?????
To run this recipe, do the following:
> python train.py hparams/interleave1.yaml
With the default hyperparameters, the system employs a convolutional frontend and
an InterleaveFormer without segmentation.
The decoder is ??????
The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).
The best model is the average of the checkpoints from last 5 epochs.
The experiment file is flexible enough to support a large variety of
different systems.
Authors
 * Yiqi Wang, 2022
"""
import numpy as np
import functools
import operator
import os
import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import logging
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.alignment.ctc_segmentation import CTCSegmentation
import gc
logger = logging.getLogger(__name__)

def check_seg(audio, text, key = None):
    valid = []
    assert len(audio) == len(text)
    for idx in range(len(audio)):
        if len(set(audio[idx])) == len(set(text[idx])):
            valid.append(idx)
    return valid

# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)
        # feats shape:      batch x max_audio_len x feature_dim e.g.: 32, 1672, 80
        # tokens_bos shape: batch x max_text_len

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)
        
        src = self.modules.CNN(feats) 
        
        if self.gen_pkl:
            # forward modules
            audio_stats, token_stats = self.get_batch_stats(batch)
            # token_stats don't have bos at front

            # get relative len
            max_abs_len = np.max(audio_stats, axis = 1).astype("float64")
            audio_stats = audio_stats.astype("float64") / max_abs_len.reshape(-1,1)
            # true_len = src.shape[1] * wav_lens.detach().cpu().numpy().reshape(-1, 1)
            true_len = np.ceil( src.shape[1] * wav_lens.detach().cpu().numpy() ).reshape(-1, 1)
            audio_stats = (audio_stats * true_len).astype(int)

            # update each segment with extra bos
            batch_token_seg_bos, batch_token_seg_eos, batch_tokens_eos_len, batch_tokens_len, text_stats = self.segment_eos_bos(tokens_bos, token_stats.copy())

            # create dict to contain all stats
            stats_dict = {}
            stats_dict["audio_stats"] = audio_stats
            stats_dict["text_stats"] = text_stats
            valid = check_seg(audio_stats, text_stats)
            audio_stats = audio_stats[valid]
            text_stats = text_stats[valid]
            stats_dict["batch_token_seg_bos"] = batch_token_seg_bos
            stats_dict["batch_token_seg_eos"] = batch_token_seg_eos
            stats_dict["batch_tokens_eos_len"] = batch_tokens_eos_len
            stats_dict["batch_tokens_len"] = batch_tokens_len
            stats_dict["valid"] = valid
            
            batch_token_seg_bos = batch_token_seg_bos[valid]
            batch_token_seg_eos = batch_token_seg_eos[valid]
            batch_tokens_eos_len = batch_tokens_eos_len[valid]
            batch_tokens_len = batch_tokens_len[valid]
            src = src[valid]
            

            # save to batch_stats, which is a list of dicts. This will be used to store stats and read from pkl
            if stage == sb.Stage.TRAIN:
                self.train_batch_stats[str(batch.id)] = stats_dict
            elif stage == sb.Stage.VALID:
                self.valid_batch_stats[str(batch.id)] = stats_dict
            elif stage == sb.Stage.TEST:
                self.test_batch_stats[str(batch.id)] = stats_dict
        
        else:
            # read batch stats from pkl
            if stage == sb.Stage.TRAIN:
                # print(self.train_batch_stats.keys())
                stats_dict = self.train_batch_stats[str(batch.id)]
            elif stage == sb.Stage.VALID:
                stats_dict = self.valid_batch_stats[str(batch.id)]
            elif stage == sb.Stage.TEST:
                stats_dict = self.test_batch_stats[str(batch.id)]
            
            audio_stats = stats_dict["audio_stats"]
            text_stats = stats_dict["text_stats"]
            valid =  stats_dict["valid"]
            # valid = check_seg(audio_stats, text_stats)
            # audio_stats = audio_stats[valid]
            # text_stats = text_stats[valid]
            batch_token_seg_bos = stats_dict["batch_token_seg_bos"][valid]
            batch_token_seg_eos = stats_dict["batch_token_seg_eos"][valid]
            batch_tokens_eos_len = stats_dict["batch_tokens_eos_len"][valid]
            batch_tokens_len = stats_dict["batch_tokens_len"][valid]
            src = src[valid]
            
        
        # batch_tokens_len is  batch.tokens_len
        seg_stats = [audio_stats, text_stats]

        assert len(set(audio_stats[0])) < 20, f"num seg > 20, audio_stats[0]"

        # enc_out is the audio representation + text representation
#         encoded_output, modalities = self.modules.Transformer(
#             src, batch_token_seg_bos, seg_stats = seg_stats, pad_idx=self.hparams.pad_index,
#         )

#         dim = encoded_output.shape[2]
#         audio_representation = []
#         text_representation = []

 
#         for i, encoded in enumerate(encoded_output):
#             a = encoded[modalities[i]==1]
#             t = encoded[modalities[i]==2]
#             audio_representation.append(a)
#             text_representation.append(t)

#         audio_representation = pad_sequence( audio_representation, batch_first=True )
#         text_representation = pad_sequence( text_representation, batch_first=True )

#         # output layer for ctc log-probabilities
#         logits = self.modules.ctc_lin(audio_representation)
#         p_ctc = self.hparams.log_softmax(logits)

#         # output layer for seq2seq log-probabilities
#         pred = self.modules.seq_lin(text_representation)
#         p_seq = self.hparams.log_softmax(pred)

#         # Compute outputs
#         hyps = None
#         if stage == sb.Stage.TRAIN:
#             hyps = None
#         elif stage == sb.Stage.VALID:
#             hyps = None
#             current_epoch = self.hparams.epoch_counter.current
#             if current_epoch % self.hparams.valid_search_interval == 0:
#                 # for the sake of efficiency, we only perform beamsearch with limited capacity
#                 # and no LM to give user some idea of how the AM is doing
                
#                 # be aware, it is src that is put into the search. 
#                 # since InterleaveFormer now inefficiently recompute everything 
#                 hyps, _ = self.hparams.valid_search(src, batch_token_seg_bos, seg_stats = seg_stats)
#         elif stage == sb.Stage.TEST:
#             hyps, _ = self.hparams.test_search( src, batch_token_seg_bos, seg_stats = seg_stats)
        
        return None # p_ctc, p_seq, wav_lens, hyps, batch_token_seg_eos, torch.tensor( batch_tokens_eos_len).to(encoded_output.device)


    def segment_eos_bos(self, token_bos, text_stats):
        """input token_bos as example. Reformulate token_bos, token_eos, text_stats"""
        batch_size, _ = text_stats.shape
        BOS = torch.tensor([1]).to(token_bos.device)
        EOS = torch.tensor([2]).to(token_bos.device)
        token_bos = token_bos[:,1:] # we remove bos from token_bos
        batch_token_seg_bos = []
        batch_token_seg_eos = []

        batch_tokens_eos_len = []
        batch_tokens_len = []
        for sample_idx in range(batch_size):
            seg_len = len( set(text_stats[sample_idx]) )
            token_seg_bos = []
            token_seg_eos = []

            tokens_eos_len = 0
            tokens_len = 0
            for s in range(seg_len):
                start_idx = 0 # to skip the bos
                if s > 0:
                    start_idx = text_stats[sample_idx][s-1]
                    # since BOS of previous segs has been added, increase by s
                    # i.e. for segment index 2, it has 2 seg before it. Add 2
                    text_stats[sample_idx][s-1] += s
                end_idx =  text_stats[sample_idx][s]
                text_seg = token_bos[sample_idx][start_idx:end_idx]
                # for token_bos
                token_seg_bos.append(BOS)
                token_seg_bos.append(text_seg)
                # for token_eos
                token_seg_eos.append(text_seg)
                token_seg_eos.append(EOS)
                # len calculation for compute objective
                text_len = len(text_seg)
                tokens_eos_len += (text_len + 1)
                tokens_len += text_len
                # if last segment, update all future segment
                # i.e. if seg index 3 is the last one. Then it has to add 4 instead of 3
                # since all previouus 3 segment include itself BOS has been added.
                if s == (seg_len - 1):
                    text_stats[sample_idx][s] += (s+1)
                    text_stats[sample_idx][s:] = text_stats[sample_idx][s]
            # assert False, f"{token_seg_bos[0].shape} {token_seg_bos[1].shape} {token_seg_bos[2].shape}"
            batch_token_seg_bos.append( torch.cat(token_seg_bos, dim = 0))
            batch_token_seg_eos.append( torch.cat(token_seg_eos, dim = 0))
            batch_tokens_eos_len.append(tokens_eos_len)
            batch_tokens_len.append(tokens_len)

        batch_tokens_eos_len = np.array(batch_tokens_eos_len)
        batch_tokens_len = np.array(batch_tokens_len)

        batch_token_seg_bos = pad_sequence( batch_token_seg_bos, batch_first=True )
        batch_token_seg_eos = pad_sequence( batch_token_seg_eos, batch_first=True )
        return batch_token_seg_bos, batch_token_seg_eos, batch_tokens_eos_len, batch_tokens_len, text_stats

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, hyps, tokens_eos, tokens_eos_lens) = predictions

        ids = batch.id
        # tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum()

        # now as training progresses we use real prediction from the prev step instead of teacher forcing

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        ).sum()

        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                # print("Pred:", predicted_words[0][:10])
                # print("Targ:", target_words[0][:10])
                self.wer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss
    
    def get_batch_stats(self, batch):

        def get_stats(seg_points):
            seg_lens = [len(l) for l in seg_points]
            max_len = max(seg_lens)
            stats = [seg + [seg[-1]]*(max_len-seg_lens[i]) \
                                    for i, seg in enumerate(seg_points)]
            return np.array(stats)

        # audio stats
        audio_stats = get_stats(batch.audio_seg_points)

        # token stats
        token_stats = get_stats(batch.token_seg_points)
        
        return audio_stats, token_stats

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0
        loss = torch.tensor(0)
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                # loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # self.scaler.scale(loss / self.grad_accumulation_factor).backward()
            if should_step:
                pass
#                 self.scaler.unscale_(self.optimizer)
#                 if self.check_gradients(loss):
#                     self.scaler.step(self.optimizer)
#                 self.scaler.update()
#                 self.optimizer_step += 1

#                 # anneal lr every update
#                 self.hparams.noam_annealing(self.optimizer)
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            # loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # (loss / self.grad_accumulation_factor).backward()
            if should_step:
                pass
#                 if self.check_gradients(loss):
#                     self.optimizer.step()
#                 self.optimizer.zero_grad()
#                 self.optimizer_step += 1

#                 # anneal lr every update
#                 self.hparams.noam_annealing(self.optimizer)
        del batch
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        del batch
        del predictions
        gc.collect()
        torch.cuda.empty_cache()
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()


def dataio_prepare(hparams, have_pkl=True):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )
    test_data = [i for k, i in test_datasets.items()]
    datasets = [train_data, valid_data] + test_data
    valtest_datasets = [valid_data] + test_data

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if hparams["speed_perturb"]:
            sig = sb.dataio.dataio.read_audio(wav)
            # factor = np.random.uniform(0.95, 1.05)
            # sig = resample(sig.numpy(), 16000, int(16000*factor))
            speed = sb.processing.speech_augmentation.SpeedPerturb(
                16000, [x for x in range(95, 105)]
            )
            sig = speed(sig.unsqueeze(0)).squeeze(0)  # torch.from_numpy(sig)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    
# Generate segments if no stats pkl saved(Ours)
    # load an ASR model and CTC aligner
    if sum(have_pkl) < 3:
        pre_trained = "speechbrain/asr-transformer-transformerlm-librispeech"
        asr_model = EncoderDecoderASR.from_hparams(source=pre_trained)
        aligner = CTCSegmentation(asr_model, kaldi_style_text=False)

        def get_audio_seg(wav, wrd):
            # Get aligned timestamps by CTC aligner
            text = [" "+ sing_wrd + " " for sing_wrd in wrd.split(" ")]
            segs = aligner(wav, text)
            intervals = segs.segments
            time_end_array = np.array(list(list(zip(*intervals))[1]))
            
            # deal with audio length < 2 sec, only one segment
            if max(time_end_array) <= 2:
                return [round(max(time_end_array)*16000)+1], [text]
            
            rounded_end = np.floor(max(time_end_array))
            max_end = int(rounded_end) + 1 if rounded_end % 2 == 0 else int(rounded_end)
            interval_range = list(range(2, max_end, 2))
            seg_index = [np.searchsorted(time_end_array, interval, side='right') - 1 for interval in interval_range]
            
            # if the last bin only has one word, we want to merge it to the previous one
            if seg_index[-1] == len(time_end_array) - 2:
                seg_index[-1] = len(time_end_array) - 1
            else:
                seg_index.append(len(time_end_array) - 1)
            
            seg_points_in_time = time_end_array[seg_index]
            audio_seg_points = [round(seg_point*16000)+1 for seg_point in seg_points_in_time]
            seg_pts = [index + 1 for index in seg_index]
            seg_pts.insert(0, 0)
            binned_wrds = [text[seg_pts[i]:seg_pts[i+1]] for i in range(len(seg_pts)-1)]
            return audio_seg_points, binned_wrds

        def get_token_seg(wrds):
            token_seg = [tokenizer.encode_as_ids(' '.join(bin_wrd)) for bin_wrd in wrds]
            return [len(functools.reduce(operator.iconcat, token_seg[:i+1], [])) for i in range(len(token_seg))]


        @sb.utils.data_pipeline.takes("wav", "wrd")
        @sb.utils.data_pipeline.provides("audio_seg_points", "token_seg_points")

        def generate_segments(wav, wrd):
            audio_seg_points, binned_wrds  = get_audio_seg(wav, wrd)
            yield audio_seg_points
            token_seg_points = get_token_seg(binned_wrds)
            yield token_seg_points

        if have_pkl[0] == 0:
            sb.dataio.dataset.add_dynamic_item([train_data], generate_segments)
        if have_pkl[1] == 0:
            sb.dataio.dataset.add_dynamic_item([valid_data], generate_segments)
        if have_pkl[2] == 0:
            sb.dataio.dataset.add_dynamic_item(test_data, generate_segments)

    # 4. Set output:
    if sum(have_pkl) == 3:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"]
        )
    else:
        if have_pkl[0] == 0:
            sb.dataio.dataset.set_output_keys(
                [train_data], ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens",
                "audio_seg_points", "token_seg_points"]
            )
        else:
            sb.dataio.dataset.set_output_keys(
                [train_data], ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"]
            )
        if have_pkl[1] == 0:
            sb.dataio.dataset.set_output_keys(
                [valid_data], ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens",
                "audio_seg_points", "token_seg_points"]
            )
        else:
            sb.dataio.dataset.set_output_keys(
                [valid_data], ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"]
            )
        if have_pkl[2] == 0:
            sb.dataio.dataset.set_output_keys(
                test_data, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens",
                "audio_seg_points", "token_seg_points"]
            )
        else:
            sb.dataio.dataset.set_output_keys(
                test_data, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"]
            )
        

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["data_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": hparams["train_csv"],
            "skip_prep": hparams["skip_prep"],
        },
    )
    
    # Check if stats pkl's are in data folder
    have_pkl = [0, 0, 0]
    data_folder = hparams["data_folder"]
    if Path(os.path.join(data_folder, 'train_batch_stats.pkl')).is_file():
        have_pkl[0] = 1

    if Path(os.path.join(data_folder, 'valid_batch_stats.pkl')).is_file():
        have_pkl[1] = 1

    if Path(os.path.join(data_folder, 'test_batch_stats.pkl')).is_file():
        have_pkl[2] = 1
        
    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams, have_pkl=have_pkl)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        data_folder=hparams["data_folder"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )
    # _fit_train of asr_brain will call fit batch!
    # _fit_valid of asr_brain will call evaluate batch!

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k],
            max_key="ACC",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )