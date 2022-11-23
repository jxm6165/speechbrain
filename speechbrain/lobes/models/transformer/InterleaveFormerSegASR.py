"""Transformer for ASR in the SpeechBrain sytle.
Authors
* Yiqi Wang, 2022
"""

import torch  # noqa 42
from torch import nn
from typing import Optional
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import ModuleList
from speechbrain.lobes.models.transformer.InterleaveFormerSeg import (
    InterleaveFormerInterface,
    rebatch,
    mask,
    unmask,
    NormalizedEmbedding,
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
    def decode(self, tgt, src, wave_len, enc_len=None):
        """This method implements a decoding step for the InterleaveFormer model.
        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        src : torch.Tensor
            Raw audio instead of encoded audio.
        enc_len : torch.LongTensor
            Not used. 
            The actual length of encoder states.
        """
        assert False, f"Need to rewrite this"
        if src.ndim == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        _, max_audio, _ = src.shape
        bin_width, max_text, = tgt.shape # bin_width x 1 becaues each time 1 token but consider bin_width # of possibility in beam search
        seg_stats = [max_audio, max_text]

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask, # this one could be the hopping causal mask! Postponed right now.
        ) = self.make_masks(src, tgt, wave_len, seg_stats)

        src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        if self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)  # add the encodings here
            pos_embs_encoder = None


        tgt = self.custom_tgt_module(tgt)
        if self.attention_type == "RelPosMHAXL":
            # we use fixed positional encodings in the decoder
            assert False, f"Not supported yet"
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)  # add the encodings here
            pos_embs_target = None
            pos_embs_encoder = None

        # souce has batch_size (which is 1) x horizon x feature where tgt is bin_size x horizon
        # Make them match in the first dimension! 
        final_src = torch.cat([src.repeat(bin_width,1,1), tgt], dim = 1)
        # assert False, f"wave: {wave_len} {src_key_padding_mask.shape} {tgt.shape} {tgt_key_padding_mask.shape} {tgt_mask.shape} "
        final_padding_mask = torch.cat([src_key_padding_mask.repeat(bin_width,1), tgt_key_padding_mask], dim = 1)

        # encoded_output is bi-modality learned representation.
        encoded_output, attn = self.encoder(
            src=final_src,
            seg_stats = seg_stats, # used by modality expert
            src_mask=tgt_mask, # this must be a causal mask, hopping style
            src_key_padding_mask=final_padding_mask,
            pos_embs=pos_embs_encoder,
        )
        prediction = encoded_output[:, max_audio: ]
        return prediction, attn[-1]

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
