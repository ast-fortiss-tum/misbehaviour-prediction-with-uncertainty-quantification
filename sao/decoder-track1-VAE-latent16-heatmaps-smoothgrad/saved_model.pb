š
ŃŁ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ÇĐ
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
{
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź*
shared_namedense_2/kernel
t
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*!
_output_shapes
:Ź*
dtype0
r
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź*
shared_namedense_2/bias
k
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes

:Ź*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ů
valueĎBĚ BĹ
˝
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
 

	0

1
2
3

	0

1
2
3
­
layer_regularization_losses
layer_metrics
regularization_losses
metrics

layers
trainable_variables
	variables
non_trainable_variables
 
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1
­
layer_regularization_losses
layer_metrics
regularization_losses
metrics

layers
trainable_variables
	variables
non_trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
layer_regularization_losses
 layer_metrics
regularization_losses
!metrics

"layers
trainable_variables
	variables
#non_trainable_variables
 
 
 

0
1
2
 
 
 
 
 
 
 
 
 
 
 
}
serving_default_z_samplingPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
ů
StatefulPartitionedCallStatefulPartitionedCallserving_default_z_samplingdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_7354
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Š
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_7526
Ô
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_7548˛
Ě

A__inference_decoder_layer_call_and_return_conditional_losses_7378

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityŚ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/MatMulĽ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp˘
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/Relu¨
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*!
_output_shapes
:Ź*
dtype02
dense_2/MatMul/ReadVariableOpĄ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
dense_2/MatMulŚ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:Ź*
dtype02 
dense_2/BiasAdd/ReadVariableOpŁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
dense_2/BiasAdd{
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
dense_2/SigmoidĆ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp¨
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
dense_1/kernel/Regularizer/Abs
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constˇ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/xź
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muli
IdentityIdentitydense_2/Sigmoid:y:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙:::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


"__inference_signature_wrapper_7354

z_sampling
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallď
StatefulPartitionedCallStatefulPartitionedCall
z_samplingunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_71752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
z_sampling
§

&__inference_decoder_layer_call_fn_7333

z_sampling
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall
z_samplingunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_73222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
z_sampling

ě
A__inference_decoder_layer_call_and_return_conditional_losses_7322

inputs
dense_1_7305
dense_1_7307
dense_2_7310
dense_2_7312
identity˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_7305dense_1_7307*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_71962!
dense_1/StatefulPartitionedCall­
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_7310dense_2_7312*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_72232!
dense_2/StatefulPartitionedCallŹ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_7305*
_output_shapes
:	*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp¨
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
dense_1/kernel/Regularizer/Abs
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constˇ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/xź
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulÂ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˙
˘
__inference__traced_save_7526
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a13ace53e88842408a984fb0c87f880c/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameý
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slicesć
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*>
_input_shapes-
+: :	::Ź:Ź: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::'#
!
_output_shapes
:Ź:"

_output_shapes

:Ź:

_output_shapes
: 
ľ
Š
A__inference_dense_1_layer_call_and_return_conditional_losses_7196

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Reluž
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp¨
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
dense_1/kernel/Regularizer/Abs
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constˇ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/xź
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ˇ
Š
A__inference_dense_2_layer_call_and_return_conditional_losses_7471

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:Ź*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:Ź*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2	
BiasAddc
SigmoidSigmoidBiasAdd:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2	
Sigmoida
IdentityIdentitySigmoid:y:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ť

i
__inference_loss_fn_0_7491:
6dense_1_kernel_regularizer_abs_readvariableop_resource
identityÖ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp¨
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
dense_1/kernel/Regularizer/Abs
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constˇ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/xź
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mule
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:


&__inference_decoder_layer_call_fn_7415

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_72892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
×

__inference__wrapped_model_7175

z_sampling2
.decoder_dense_1_matmul_readvariableop_resource3
/decoder_dense_1_biasadd_readvariableop_resource2
.decoder_dense_2_matmul_readvariableop_resource3
/decoder_dense_2_biasadd_readvariableop_resource
identityž
%decoder/dense_1/MatMul/ReadVariableOpReadVariableOp.decoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02'
%decoder/dense_1/MatMul/ReadVariableOp¨
decoder/dense_1/MatMulMatMul
z_sampling-decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder/dense_1/MatMul˝
&decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02(
&decoder/dense_1/BiasAdd/ReadVariableOpÂ
decoder/dense_1/BiasAddBiasAdd decoder/dense_1/MatMul:product:0.decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder/dense_1/BiasAdd
decoder/dense_1/ReluRelu decoder/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder/dense_1/ReluŔ
%decoder/dense_2/MatMul/ReadVariableOpReadVariableOp.decoder_dense_2_matmul_readvariableop_resource*!
_output_shapes
:Ź*
dtype02'
%decoder/dense_2/MatMul/ReadVariableOpÁ
decoder/dense_2/MatMulMatMul"decoder/dense_1/Relu:activations:0-decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
decoder/dense_2/MatMulž
&decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes

:Ź*
dtype02(
&decoder/dense_2/BiasAdd/ReadVariableOpĂ
decoder/dense_2/BiasAddBiasAdd decoder/dense_2/MatMul:product:0.decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
decoder/dense_2/BiasAdd
decoder/dense_2/SigmoidSigmoid decoder/dense_2/BiasAdd:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
decoder/dense_2/Sigmoidq
IdentityIdentitydecoder/dense_2/Sigmoid:y:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙:::::S O
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
z_sampling
Ě

A__inference_decoder_layer_call_and_return_conditional_losses_7402

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityŚ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/MatMulĽ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp˘
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/Relu¨
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*!
_output_shapes
:Ź*
dtype02
dense_2/MatMul/ReadVariableOpĄ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
dense_2/MatMulŚ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:Ź*
dtype02 
dense_2/BiasAdd/ReadVariableOpŁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
dense_2/BiasAdd{
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
dense_2/SigmoidĆ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp¨
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
dense_1/kernel/Regularizer/Abs
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constˇ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/xź
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muli
IdentityIdentitydense_2/Sigmoid:y:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙:::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¤
đ
A__inference_decoder_layer_call_and_return_conditional_losses_7266

z_sampling
dense_1_7249
dense_1_7251
dense_2_7254
dense_2_7256
identity˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCall
z_samplingdense_1_7249dense_1_7251*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_71962!
dense_1/StatefulPartitionedCall­
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_7254dense_2_7256*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_72232!
dense_2/StatefulPartitionedCallŹ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_7249*
_output_shapes
:	*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp¨
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
dense_1/kernel/Regularizer/Abs
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constˇ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/xź
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulÂ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:S O
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
z_sampling
¤
đ
A__inference_decoder_layer_call_and_return_conditional_losses_7246

z_sampling
dense_1_7207
dense_1_7209
dense_2_7234
dense_2_7236
identity˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCall
z_samplingdense_1_7207dense_1_7209*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_71962!
dense_1/StatefulPartitionedCall­
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_7234dense_2_7236*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_72232!
dense_2/StatefulPartitionedCallŹ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_7207*
_output_shapes
:	*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp¨
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
dense_1/kernel/Regularizer/Abs
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constˇ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/xź
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulÂ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:S O
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
z_sampling
§

&__inference_decoder_layer_call_fn_7300

z_sampling
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall
z_samplingunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_72892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
z_sampling

ě
A__inference_decoder_layer_call_and_return_conditional_losses_7289

inputs
dense_1_7272
dense_1_7274
dense_2_7277
dense_2_7279
identity˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_7272dense_1_7274*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_71962!
dense_1/StatefulPartitionedCall­
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_7277dense_2_7279*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_72232!
dense_2/StatefulPartitionedCallŹ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_7272*
_output_shapes
:	*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp¨
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
dense_1/kernel/Regularizer/Abs
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constˇ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/xź
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulÂ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


&__inference_decoder_layer_call_fn_7428

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_73222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ü
{
&__inference_dense_2_layer_call_fn_7480

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_72232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ř
{
&__inference_dense_1_layer_call_fn_7460

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallň
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_71962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ť
Ž
 __inference__traced_restore_7548
file_prefix#
assignvariableop_dense_1_kernel#
assignvariableop_1_dense_1_bias%
!assignvariableop_2_dense_2_kernel#
assignvariableop_3_dense_2_bias

identity_5˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slicesÄ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ś
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpş

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4Ź

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ľ
Š
A__inference_dense_1_layer_call_and_return_conditional_losses_7451

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Reluž
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp¨
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
dense_1/kernel/Regularizer/Abs
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constˇ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/xź
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ˇ
Š
A__inference_dense_2_layer_call_and_return_conditional_losses_7223

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:Ź*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:Ź*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2	
BiasAddc
SigmoidSigmoidBiasAdd:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2	
Sigmoida
IdentityIdentitySigmoid:y:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*˛
serving_default
A

z_sampling3
serving_default_z_sampling:0˙˙˙˙˙˙˙˙˙=
dense_22
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙Źtensorflow/serving/predict:a

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
regularization_losses
trainable_variables
	variables
	keras_api

signatures
*$&call_and_return_all_conditional_losses
%__call__
&_default_save_signature"ö
_tf_keras_networkÚ{"class_name": "Functional", "name": "decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z_sampling"}, "name": "z_sampling", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["z_sampling", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 38400, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["z_sampling", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z_sampling"}, "name": "z_sampling", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["z_sampling", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 38400, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["z_sampling", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
ń"î
_tf_keras_input_layerÎ{"class_name": "InputLayer", "name": "z_sampling", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z_sampling"}}
Ş

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
*'&call_and_return_all_conditional_losses
(__call__"
_tf_keras_layerë{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
ř

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*)&call_and_return_all_conditional_losses
*__call__"Ó
_tf_keras_layerš{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 38400, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
'
+0"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
Ę
layer_regularization_losses
layer_metrics
regularization_losses
metrics

layers
trainable_variables
	variables
non_trainable_variables
%__call__
&_default_save_signature
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
,
,serving_default"
signature_map
!:	2dense_1/kernel
:2dense_1/bias
'
+0"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
­
layer_regularization_losses
layer_metrics
regularization_losses
metrics

layers
trainable_variables
	variables
non_trainable_variables
(__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
#:!Ź2dense_2/kernel
:Ź2dense_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
layer_regularization_losses
 layer_metrics
regularization_losses
!metrics

"layers
trainable_variables
	variables
#non_trainable_variables
*__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ň2Ď
A__inference_decoder_layer_call_and_return_conditional_losses_7402
A__inference_decoder_layer_call_and_return_conditional_losses_7266
A__inference_decoder_layer_call_and_return_conditional_losses_7246
A__inference_decoder_layer_call_and_return_conditional_losses_7378Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ć2ă
&__inference_decoder_layer_call_fn_7415
&__inference_decoder_layer_call_fn_7333
&__inference_decoder_layer_call_fn_7300
&__inference_decoder_layer_call_fn_7428Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ŕ2Ý
__inference__wrapped_model_7175š
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *)˘&
$!

z_sampling˙˙˙˙˙˙˙˙˙
ë2č
A__inference_dense_1_layer_call_and_return_conditional_losses_7451˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Đ2Í
&__inference_dense_1_layer_call_fn_7460˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ë2č
A__inference_dense_2_layer_call_and_return_conditional_losses_7471˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Đ2Í
&__inference_dense_2_layer_call_fn_7480˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ą2Ž
__inference_loss_fn_0_7491
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
4B2
"__inference_signature_wrapper_7354
z_sampling
__inference__wrapped_model_7175p	
3˘0
)˘&
$!

z_sampling˙˙˙˙˙˙˙˙˙
Ş "3Ş0
.
dense_2# 
dense_2˙˙˙˙˙˙˙˙˙Źą
A__inference_decoder_layer_call_and_return_conditional_losses_7246l	
;˘8
1˘.
$!

z_sampling˙˙˙˙˙˙˙˙˙
p

 
Ş "'˘$

0˙˙˙˙˙˙˙˙˙Ź
 ą
A__inference_decoder_layer_call_and_return_conditional_losses_7266l	
;˘8
1˘.
$!

z_sampling˙˙˙˙˙˙˙˙˙
p 

 
Ş "'˘$

0˙˙˙˙˙˙˙˙˙Ź
 ­
A__inference_decoder_layer_call_and_return_conditional_losses_7378h	
7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "'˘$

0˙˙˙˙˙˙˙˙˙Ź
 ­
A__inference_decoder_layer_call_and_return_conditional_losses_7402h	
7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "'˘$

0˙˙˙˙˙˙˙˙˙Ź
 
&__inference_decoder_layer_call_fn_7300_	
;˘8
1˘.
$!

z_sampling˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙Ź
&__inference_decoder_layer_call_fn_7333_	
;˘8
1˘.
$!

z_sampling˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙Ź
&__inference_decoder_layer_call_fn_7415[	
7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙Ź
&__inference_decoder_layer_call_fn_7428[	
7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙Ź˘
A__inference_dense_1_layer_call_and_return_conditional_losses_7451]	
/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 z
&__inference_dense_1_layer_call_fn_7460P	
/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙¤
A__inference_dense_2_layer_call_and_return_conditional_losses_7471_0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "'˘$

0˙˙˙˙˙˙˙˙˙Ź
 |
&__inference_dense_2_layer_call_fn_7480R0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ź9
__inference_loss_fn_0_7491	˘

˘ 
Ş " ¤
"__inference_signature_wrapper_7354~	
A˘>
˘ 
7Ş4
2

z_sampling$!

z_sampling˙˙˙˙˙˙˙˙˙"3Ş0
.
dense_2# 
dense_2˙˙˙˙˙˙˙˙˙Ź