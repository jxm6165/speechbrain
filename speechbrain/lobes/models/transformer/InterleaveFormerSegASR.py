"""Transformer for ASR in the SpeechBrain sytle.
Authors
* Yiqi Wang, 2022
"""

import torch  # noqa 42
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import ModuleList
from speechbrain.lobes.models.transformer.InterleaveFormerSeg import (
    InterleaveFormerInterface,
    rebatch,
    mask,
    unmask,
    NormalizedEmbedding,
    get_lookahead_hopping_mask
)
from speechbrain.nnet.activations import Swish
import logging
# from speechbrain.dataio.dataio import length_to_mask

logger = logging.getLogger(__name__)

class InterleaveFormerASR(InterleaveFormerInterface):
    """This is an implementation of InterleaveFormer model for ASR.
    The architecture is based on the paper "PLACE HODLER":
    arxiv PLACE HODLER
    Arguments
    ----------
    tgt_vocab: int
        Size of vocabulary.
    input_size: int
        Input feature size.
    d_model : int, optional
        Embedding dimension size.
        (default=512).
    nhead : int, optional
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers : int, optional
        The number of causal-encoder-layers (i.e. decoder) in the InterleaveFormer (default=6).
    num_decoder_layers : int, optional
        No actual decoder is needed.
    dim_ffn : int, optional
        The dimension of the feedforward network model (default=2048).
    dropout : int, optional
        The dropout value (default=0.1).
    activation : torch.nn.Module, optional
        The activation function of FFN layers.
        Recommended: relu or gelu (default=relu).
    positional_encoding: str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine' for fixed absolute positional encodings.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size: int, optional
        Kernel size in convolutional layers when Conformer is used.
    bias: bool, optional
        Whether to use bias in Conformer convolutional layers.
    encoder_module: str, optional
        InterleaveFormer as a causal encoder. No other option!
    conformer_activation: torch.nn.Module, optional
        NOT USED
        Activation module used after Conformer convolutional layers. E.g. Swish, ReLU etc. it has to be a torch Module.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    max_length: int, optional
        Max length for the target and source sequence in input.
        Used for positional encodings.
    causal: bool, optional
        Whether the encoder should be causal or not (InterleaveFormer is always causal).
        If causal the Conformer convolutional layer is causal.
    Example
    -------
    >>> src = torch.rand([8, 200, 512]) # 200 is the padded total length including many bi-modality segments
    >>> tgt = torch.randint(0, 720, [8, 200])
    >>> net = TransformerASR(
    ...     720, 512, 512, 8, 1, 1, 1024, activation=torch.nn.GELU
    ... )
    >>> enc_out = net.forward(src, tgt) # not that enc_out actually contains both audio and text
    >>> enc_out.shape
    torch.Size([8, 200, 512])
    """

    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=0,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "InterleaveFormer",
        conformer_activation: Optional[nn.Module] = Swish,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
        )

        self.nhead = nhead # save it for causal mask broadcasting 

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )
        self.custom_tgt_module = ModuleList(
            NormalizedEmbedding(d_model, tgt_vocab)
        )

        # reset parameters using xavier_normal_
        self._init_params()

    def forward(self, src, tgt, seg_stats, pad_idx=0):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the causal encoder.
        tgt : torch.Tensor
            The sequence to the causal encoder.
        seg_stats:
            a list of 2 np.array describe segment info.
            audio_stats: numpy.array
                array with size: batch x max_seg_num. Each element is the end idx of an aduio segment within a sequence
            text_stats: numpy.array
                array with size: batch x max_seg_num. Each element is the end idx of a text segment within a sequence
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """
        audio_stats, text_stats = seg_stats
        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.ndim == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)
        
        # audio embedding
        src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        if self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)  # add the encodings here
            pos_embs_encoder = None

        # text embedding
        tgt = self.custom_tgt_module(tgt)
        if self.attention_type == "RelPosMHAXL":
            assert False, f"Don't support RelPosMHAXL yet"
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_target = None
            pos_embs_encoder = None

        # rebatching: interleaving and repadding
        rebatch_sample, modalities = rebatch(src, tgt, audio_stats, text_stats)
        # print( f"{rebatch_sample.shape} {src.shape}" )
        (
            padding_mask,
            causal_mask, # causal for text, non-causal for audio.
        ) = self.make_masks(rebatch_sample, modalities, audio_stats, text_stats)
        # repeat each sample's causal mask by num_heads # of times.
        causal_mask = torch.cat( causal_mask ).repeat_interleave(repeats = self.nhead, dim = 0).to(rebatch_sample.device)
        padding_mask = padding_mask.to(rebatch_sample.device)
        
        # encoded_output is bi-modality learned representation.
        encoded_output, _ = self.encoder(
            src= rebatch_sample,
            modalities = modalities, # used by modality expert
            src_mask= causal_mask, # this must be a causal mask, hopping style
            src_key_padding_mask=padding_mask,
            pos_embs=pos_embs_encoder,
        )

        return encoded_output, modalities

    def make_masks(self, rebatch_sample, modalities, audio_stats, text_stats):
        """
        This method generates the masks for training the transformer model.
        1. create a normal causal mask
        2. for current segment j, unmask all audio frames belongs to j
        3. if j > 0, then mask audio frames belongs to segment j for all future segment j' > j      
        Arguments
        ---------
        rebatch_sample: tensor
            bi-modality samples that are interleaved and repadded.
        modalities : tensor
            Each element indicates the modality. 1 for audio. 2 for text. 0 for padding.
            size: batch x max_sample_len. 
        audio_stats: numpy.array, optional
            array with size: batch x max_seg_num. Each element is the end idx of an aduio segment within a sequence
        text_stats:
            array with size: batch x max_seg_num. Each element is the end idx of a text segment within a sequence
        """
        batch_size, max_len, dim = rebatch_sample.shape
        
        # make a normal causal mask first
        inf = float( 'inf')
        normal_causal_mask = torch.tensor( - inf).repeat(max_len, max_len)
        normal_causal_mask = torch.triu(normal_causal_mask,1)

        # make causal mask for each sample 
        final_mask = []
        for _ in range(0, batch_size):
            sample_idx = _
            hopping_causal_mask = normal_causal_mask.clone()
            # number segment == unique element in an array
            num_seg = len( set( audio_stats[sample_idx] ) )
            
            for idx in range(  num_seg ):
                # do unmask operation for each audio segment
                start_idx = 0
                end_idx = audio_stats[sample_idx][idx]
                if idx > 0:
                    # consider all the past audio and text for start
                    start_idx =  audio_stats[sample_idx][idx-1] + text_stats[sample_idx][idx-1]
                    # consider all the past audio/text + current audio
                    end_idx = audio_stats[sample_idx][idx] + text_stats[sample_idx][idx-1]
                hopping_causal_mask = unmask( hopping_causal_mask, start_idx, end_idx) 
                if num_seg == 1:
                    # if there's 1 segment, do unmask but no need for masking
                    final_mask.append( torch.unsqueeze( hopping_causal_mask.clone(), 0) )
                    break
                else:
                    # do mask operation
                    delta = text_stats[sample_idx][idx]
                    if idx > 0:
                        delta -= text_stats[sample_idx][idx-1]
                    start_row_idx = end_idx + (delta )
                    if start_row_idx < len(modalities[sample_idx]):
                        # mask the past audio + bos (i.e. end_idx + 1)
                        hopping_causal_mask = mask( hopping_causal_mask, start_row_idx, start_idx, end_idx + 1) 

            if num_seg > 1:
                # for seg_num == 1, only unmask is needed for audio, mask has been appended.
                # for seg_num > 1, need unmask,mask. Append here.
                final_mask.append( torch.unsqueeze( hopping_causal_mask.clone(), 0) )
        
        padding_mask = modalities == 0
        return padding_mask, final_mask

    @torch.no_grad()
    def decode_oracle(self, history, src, memory, src_pad=None):
        """This method implements a decoding step for the InterleaveFormerSeg model.
        The segmentation is given by pre-processing, known as oracle.
        Arguments
        ---------
        history: torch.Tensor
            past transcriptions associated with past audio segments
        src : torch.Tensor
            Raw audio instead of encoded audio.
        memory: torch.Tensor
            keep track of all text inputs for current segment
        src_pad : torch.LongTensor
            padding for the src
        """

        history_len = [ len(h) for h in history]
        assert len(history_len) == len(src)

        # history emb + src + memory emb
        final_src = []
        modalities = []
        seg_point = []
        pred_range = []
        pred_lens = []
        for idx, h in enumerate(history):
            # past transcription embedding
            hist = torch.tensor(np.array(h)).to(src[0].device).unsqueeze(0)
            history_emb = self.custom_tgt_module( hist )
            if len(h) > 0:
                # print("Check h len:", len(h), h[-10:])
                if len(h) > 100:
                    # if history is super long that is exceeding max pos emb
                    history_emb = history_emb[:,-100:, :]
                history_emb = history_emb + self.positional_encoding(history_emb)
            history_emb = history_emb.squeeze(0)
            
            # present audio embedding
            audio_seg = src[idx].unsqueeze(0) 
            if audio_seg.ndim == 4:
                b , t, ch1, ch2 = audio_seg.shape
                audio_seg = audio_seg.reshape(b , t, ch1 * ch2)
            audio_seg_emb = self.custom_src_module( audio_seg )
            history_offset = torch.zeros(1,len(h), audio_seg_emb.shape[2]).to(audio_seg_emb.device)
            audio_seg_emb = ( audio_seg_emb + self.positional_encoding(
                torch.cat([ history_offset, audio_seg_emb], dim = 1)
              )[:,len(h):,:]
            ).squeeze(0)
            # present on-going transcription embedding
            m_raw = memory[idx]
            mem = torch.tensor( np.array(m_raw) ).to(audio_seg_emb.device).unsqueeze(0)
            memory_emb = self.custom_tgt_module( mem )
            # print("Mem emb:", torch.cat([history_offset, memory_emb], dim = 1)[:,len(h):,:].shape, memory_emb.shape)
            memory_emb = memory_emb + self.positional_encoding(
                torch.cat([history_offset, memory_emb], dim = 1)[:,len(h):,:]
            )
            memory_emb = memory_emb.squeeze(0)
            # final interleaved sample and its modalities
            hist_audio_mem = torch.cat( [ history_emb, audio_seg_emb, memory_emb ], dim = 0)
            mod = np.array( [1] * len(history_emb) + [2] * len(audio_seg_emb) + [1] * len(memory_emb) )
            # print("Check modality per beam:", len(history_emb), len(audio_seg_emb), len(memory_emb))
            final_src.append(hist_audio_mem )
            modalities.append( torch.tensor( mod ))
            present_start = len(history_emb) + len(audio_seg_emb)
            seg_point.append( present_start )
            present_end = present_start + len(memory_emb)
            pred_range.append(( present_start, present_end ))
            pred_lens.append(present_end - present_start)
        final_src = pad_sequence(final_src, batch_first=True).to(src[0].device)
        modalities = pad_sequence(modalities, batch_first=True).to(src[0].device)
        final_pad = (modalities == 0).to(final_src.device)
        
        # make a normal causal mask first
        causal_mask = []
        inf = float( 'inf')
        max_len = final_src.shape[1]
        normal_causal_mask = torch.tensor( - inf).repeat(max_len, max_len)
        normal_causal_mask = torch.triu(normal_causal_mask,1)
        for seg in seg_point:
            _mask = normal_causal_mask.clone()
            # do unmask operation for each audio segment and previous transcriptions
            hopping_causal_mask = unmask( _mask, 0, seg) 
            causal_mask.append( torch.unsqueeze( hopping_causal_mask.clone(), 0) )
        final_causal_mask = torch.cat( causal_mask ).repeat_interleave(repeats = self.nhead, dim = 0).to(final_src.device)

        # encoded_output is bi-modality learned representation.
        encoded_output, attn = self.encoder(
            src= final_src,
            modalities = modalities, # used by modality expert
            src_mask= final_causal_mask, # this must be a causal mask, hopping style
            src_key_padding_mask= final_pad,
            pos_embs= None,
        )

        prediction = []
        max_pred_len = max(pred_lens)

        for i, (s,e) in enumerate( pred_range ):
            # e-1 to retrieve the latest token (the last one)
            pred = encoded_output[i, e-1 ].unsqueeze(0)
            # print("PRED:", pred.shape)
            prediction.append(pred)
        return torch.cat( prediction, dim = 0),  attn[-1]

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
