��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
�
Adam/v/thicknesses/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/thicknesses/bias

+Adam/v/thicknesses/bias/Read/ReadVariableOpReadVariableOpAdam/v/thicknesses/bias*
_output_shapes
:*
dtype0
�
Adam/m/thicknesses/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/thicknesses/bias

+Adam/m/thicknesses/bias/Read/ReadVariableOpReadVariableOpAdam/m/thicknesses/bias*
_output_shapes
:*
dtype0
�
Adam/v/thicknesses/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_nameAdam/v/thicknesses/kernel
�
-Adam/v/thicknesses/kernel/Read/ReadVariableOpReadVariableOpAdam/v/thicknesses/kernel*
_output_shapes

: *
dtype0
�
Adam/m/thicknesses/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_nameAdam/m/thicknesses/kernel
�
-Adam/m/thicknesses/kernel/Read/ReadVariableOpReadVariableOpAdam/m/thicknesses/kernel*
_output_shapes

: *
dtype0
�
 Adam/v/dense_11/p_re_lu_11/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/v/dense_11/p_re_lu_11/alpha
�
4Adam/v/dense_11/p_re_lu_11/alpha/Read/ReadVariableOpReadVariableOp Adam/v/dense_11/p_re_lu_11/alpha*
_output_shapes
: *
dtype0
�
 Adam/m/dense_11/p_re_lu_11/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/m/dense_11/p_re_lu_11/alpha
�
4Adam/m/dense_11/p_re_lu_11/alpha/Read/ReadVariableOpReadVariableOp Adam/m/dense_11/p_re_lu_11/alpha*
_output_shapes
: *
dtype0
�
Adam/v/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_11/bias
y
(Adam/v/dense_11/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_11/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_11/bias
y
(Adam/m/dense_11/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_11/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/v/dense_11/kernel
�
*Adam/v/dense_11/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_11/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/m/dense_11/kernel
�
*Adam/m/dense_11/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_11/kernel*
_output_shapes

:  *
dtype0
�
 Adam/v/dense_10/p_re_lu_10/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/v/dense_10/p_re_lu_10/alpha
�
4Adam/v/dense_10/p_re_lu_10/alpha/Read/ReadVariableOpReadVariableOp Adam/v/dense_10/p_re_lu_10/alpha*
_output_shapes
: *
dtype0
�
 Adam/m/dense_10/p_re_lu_10/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/m/dense_10/p_re_lu_10/alpha
�
4Adam/m/dense_10/p_re_lu_10/alpha/Read/ReadVariableOpReadVariableOp Adam/m/dense_10/p_re_lu_10/alpha*
_output_shapes
: *
dtype0
�
Adam/v/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_10/bias
y
(Adam/v/dense_10/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_10/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_10/bias
y
(Adam/m/dense_10/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_10/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:` *'
shared_nameAdam/v/dense_10/kernel
�
*Adam/v/dense_10/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_10/kernel*
_output_shapes

:` *
dtype0
�
Adam/m/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:` *'
shared_nameAdam/m/dense_10/kernel
�
*Adam/m/dense_10/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_10/kernel*
_output_shapes

:` *
dtype0
�
Adam/v/second_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/v/second_layer/bias
�
,Adam/v/second_layer/bias/Read/ReadVariableOpReadVariableOpAdam/v/second_layer/bias*
_output_shapes
: *
dtype0
�
Adam/m/second_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/m/second_layer/bias
�
,Adam/m/second_layer/bias/Read/ReadVariableOpReadVariableOpAdam/m/second_layer/bias*
_output_shapes
: *
dtype0
�
Adam/v/second_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *+
shared_nameAdam/v/second_layer/kernel
�
.Adam/v/second_layer/kernel/Read/ReadVariableOpReadVariableOpAdam/v/second_layer/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/second_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *+
shared_nameAdam/m/second_layer/kernel
�
.Adam/m/second_layer/kernel/Read/ReadVariableOpReadVariableOpAdam/m/second_layer/kernel*
_output_shapes

:  *
dtype0
�
Adam/v/first_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/v/first_layer/bias

+Adam/v/first_layer/bias/Read/ReadVariableOpReadVariableOpAdam/v/first_layer/bias*
_output_shapes
: *
dtype0
�
Adam/m/first_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/m/first_layer/bias

+Adam/m/first_layer/bias/Read/ReadVariableOpReadVariableOpAdam/m/first_layer/bias*
_output_shapes
: *
dtype0
�
Adam/v/first_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  **
shared_nameAdam/v/first_layer/kernel
�
-Adam/v/first_layer/kernel/Read/ReadVariableOpReadVariableOpAdam/v/first_layer/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/first_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  **
shared_nameAdam/m/first_layer/kernel
�
-Adam/m/first_layer/kernel/Read/ReadVariableOpReadVariableOpAdam/m/first_layer/kernel*
_output_shapes

:  *
dtype0
�
Adam/v/dense_9/p_re_lu_9/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/v/dense_9/p_re_lu_9/alpha
�
2Adam/v/dense_9/p_re_lu_9/alpha/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/p_re_lu_9/alpha*
_output_shapes
: *
dtype0
�
Adam/m/dense_9/p_re_lu_9/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/m/dense_9/p_re_lu_9/alpha
�
2Adam/m/dense_9/p_re_lu_9/alpha/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/p_re_lu_9/alpha*
_output_shapes
: *
dtype0
~
Adam/v/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/v/dense_9/bias
w
'Adam/v/dense_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/bias*
_output_shapes
: *
dtype0
~
Adam/m/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/m/dense_9/bias
w
'Adam/m/dense_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/v/dense_9/kernel

)Adam/v/dense_9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/m/dense_9/kernel

)Adam/m/dense_9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/kernel*
_output_shapes

:  *
dtype0
�
Adam/v/dense_8/p_re_lu_8/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/v/dense_8/p_re_lu_8/alpha
�
2Adam/v/dense_8/p_re_lu_8/alpha/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/p_re_lu_8/alpha*
_output_shapes
: *
dtype0
�
Adam/m/dense_8/p_re_lu_8/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/m/dense_8/p_re_lu_8/alpha
�
2Adam/m/dense_8/p_re_lu_8/alpha/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/p_re_lu_8/alpha*
_output_shapes
: *
dtype0
~
Adam/v/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/v/dense_8/bias
w
'Adam/v/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/bias*
_output_shapes
: *
dtype0
~
Adam/m/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/m/dense_8/bias
w
'Adam/m/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/v/dense_8/kernel

)Adam/v/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/kernel*
_output_shapes

:@ *
dtype0
�
Adam/m/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/m/dense_8/kernel

)Adam/m/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/kernel*
_output_shapes

:@ *
dtype0
�
Adam/v/dense_7/p_re_lu_7/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/v/dense_7/p_re_lu_7/alpha
�
2Adam/v/dense_7/p_re_lu_7/alpha/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/p_re_lu_7/alpha*
_output_shapes
:@*
dtype0
�
Adam/m/dense_7/p_re_lu_7/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/m/dense_7/p_re_lu_7/alpha
�
2Adam/m/dense_7/p_re_lu_7/alpha/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/p_re_lu_7/alpha*
_output_shapes
:@*
dtype0
~
Adam/v/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/v/dense_7/bias
w
'Adam/v/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/bias*
_output_shapes
:@*
dtype0
~
Adam/m/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/m/dense_7/bias
w
'Adam/m/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*&
shared_nameAdam/v/dense_7/kernel
�
)Adam/v/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*&
shared_nameAdam/m/dense_7/kernel
�
)Adam/m/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/v/dense_6/p_re_lu_6/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name Adam/v/dense_6/p_re_lu_6/alpha
�
2Adam/v/dense_6/p_re_lu_6/alpha/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/p_re_lu_6/alpha*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_6/p_re_lu_6/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name Adam/m/dense_6/p_re_lu_6/alpha
�
2Adam/m/dense_6/p_re_lu_6/alpha/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/p_re_lu_6/alpha*
_output_shapes	
:�*
dtype0

Adam/v/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/dense_6/bias
x
'Adam/v/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/bias*
_output_shapes	
:�*
dtype0

Adam/m/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/dense_6/bias
x
'Adam/m/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/dense_6/kernel
�
)Adam/v/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/dense_6/kernel
�
)Adam/m/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/kernel* 
_output_shapes
:
��*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_11/p_re_lu_11/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namedense_11/p_re_lu_11/alpha
�
-dense_11/p_re_lu_11/alpha/Read/ReadVariableOpReadVariableOpdense_11/p_re_lu_11/alpha*
_output_shapes
: *
dtype0
�
dense_10/p_re_lu_10/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namedense_10/p_re_lu_10/alpha
�
-dense_10/p_re_lu_10/alpha/Read/ReadVariableOpReadVariableOpdense_10/p_re_lu_10/alpha*
_output_shapes
: *
dtype0
�
dense_9/p_re_lu_9/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namedense_9/p_re_lu_9/alpha

+dense_9/p_re_lu_9/alpha/Read/ReadVariableOpReadVariableOpdense_9/p_re_lu_9/alpha*
_output_shapes
: *
dtype0
�
dense_8/p_re_lu_8/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namedense_8/p_re_lu_8/alpha

+dense_8/p_re_lu_8/alpha/Read/ReadVariableOpReadVariableOpdense_8/p_re_lu_8/alpha*
_output_shapes
: *
dtype0
�
dense_7/p_re_lu_7/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namedense_7/p_re_lu_7/alpha

+dense_7/p_re_lu_7/alpha/Read/ReadVariableOpReadVariableOpdense_7/p_re_lu_7/alpha*
_output_shapes
:@*
dtype0
�
dense_6/p_re_lu_6/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_namedense_6/p_re_lu_6/alpha
�
+dense_6/p_re_lu_6/alpha/Read/ReadVariableOpReadVariableOpdense_6/p_re_lu_6/alpha*
_output_shapes	
:�*
dtype0
x
thicknesses/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namethicknesses/bias
q
$thicknesses/bias/Read/ReadVariableOpReadVariableOpthicknesses/bias*
_output_shapes
:*
dtype0
�
thicknesses/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *#
shared_namethicknesses/kernel
y
&thicknesses/kernel/Read/ReadVariableOpReadVariableOpthicknesses/kernel*
_output_shapes

: *
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
: *
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:  *
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
: *
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:` * 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:` *
dtype0
z
second_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namesecond_layer/bias
s
%second_layer/bias/Read/ReadVariableOpReadVariableOpsecond_layer/bias*
_output_shapes
: *
dtype0
�
second_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_namesecond_layer/kernel
{
'second_layer/kernel/Read/ReadVariableOpReadVariableOpsecond_layer/kernel*
_output_shapes

:  *
dtype0
x
first_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namefirst_layer/bias
q
$first_layer/bias/Read/ReadVariableOpReadVariableOpfirst_layer/bias*
_output_shapes
: *
dtype0
�
first_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *#
shared_namefirst_layer/kernel
y
&first_layer/kernel/Read/ReadVariableOpReadVariableOpfirst_layer/kernel*
_output_shapes

:  *
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
: *
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:  *
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
: *
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:@ *
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:@*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	�@*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:�*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
��*
dtype0
|
serving_default_input_2Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_6/kerneldense_6/biasdense_6/p_re_lu_6/alphadense_7/kerneldense_7/biasdense_7/p_re_lu_7/alphadense_8/kerneldense_8/biasdense_8/p_re_lu_8/alphadense_9/kerneldense_9/biasdense_9/p_re_lu_9/alphafirst_layer/kernelfirst_layer/biassecond_layer/kernelsecond_layer/biasdense_10/kerneldense_10/biasdense_10/p_re_lu_10/alphadense_11/kerneldense_11/biasdense_11/p_re_lu_11/alphathicknesses/kernelthicknesses/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2071755

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!
activation

"kernel
#bias*
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*
activation

+kernel
,bias*
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3_random_generator* 
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:
activation

;kernel
<bias*
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
C_random_generator* 
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
J
activation

Kkernel
Lbias*
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
S_random_generator* 
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias*
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias*
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses* 
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
p
activation

qkernel
rbias*
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
y_random_generator* 
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
�
activation
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
"0
#1
�2
+3
,4
�5
;6
<7
�8
K9
L10
�11
Z12
[13
b14
c15
q16
r17
�18
�19
�20
�21
�22
�23*
�
"0
#1
�2
+3
,4
�5
;6
<7
�8
K9
L10
�11
Z12
[13
b14
c15
q16
r17
�18
�19
�20
�21
�22
�23*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*
* 

�serving_default* 

"0
#1
�2*

"0
#1
�2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�alpha*
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1
�2*

+0
,1
�2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�alpha*
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

;0
<1
�2*

;0
<1
�2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�alpha*
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

K0
L1
�2*

K0
L1
�2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�alpha*
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

Z0
[1*

Z0
[1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEfirst_layer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEfirst_layer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

b0
c1*

b0
c1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
c]
VARIABLE_VALUEsecond_layer/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEsecond_layer/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

q0
r1
�2*

q0
r1
�2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�alpha*
_Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_10/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1
�2*

�0
�1
�2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�alpha*
_Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_11/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEthicknesses/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEthicknesses/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdense_6/p_re_lu_6/alpha&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdense_7/p_re_lu_7/alpha&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdense_8/p_re_lu_8/alpha&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEdense_9/p_re_lu_9/alpha'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_10/p_re_lu_10/alpha'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_11/p_re_lu_11/alpha'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
* 
z
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15*
<
�0
�1
�2
�3
�4
�5
�6*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23*
* 
* 
* 

!0*
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

*0*
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

:0*
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

J0*
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

p0*
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
`Z
VARIABLE_VALUEAdam/m/dense_6/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_6/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_6/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_6/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/dense_6/p_re_lu_6/alpha1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/dense_6/p_re_lu_6/alpha1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_7/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_7/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_7/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_7/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/dense_7/p_re_lu_7/alpha2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/dense_7/p_re_lu_7/alpha2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_8/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_8/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_8/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_8/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/dense_8/p_re_lu_8/alpha2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/dense_8/p_re_lu_8/alpha2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_9/kernel2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_9/kernel2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_9/bias2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_9/bias2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/dense_9/p_re_lu_9/alpha2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/dense_9/p_re_lu_9/alpha2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/first_layer/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/first_layer/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/first_layer/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/first_layer/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/second_layer/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/second_layer/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/second_layer/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/second_layer/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_10/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_10/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_10/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_10/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/dense_10/p_re_lu_10/alpha2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/dense_10/p_re_lu_10/alpha2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_11/kernel2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_11/kernel2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_11/bias2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_11/bias2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/dense_11/p_re_lu_11/alpha2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/dense_11/p_re_lu_11/alpha2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/thicknesses/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/thicknesses/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/thicknesses/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/thicknesses/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_64keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_64keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_54keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_54keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasfirst_layer/kernelfirst_layer/biassecond_layer/kernelsecond_layer/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasthicknesses/kernelthicknesses/biasdense_6/p_re_lu_6/alphadense_7/p_re_lu_7/alphadense_8/p_re_lu_8/alphadense_9/p_re_lu_9/alphadense_10/p_re_lu_10/alphadense_11/p_re_lu_11/alpha	iterationlearning_rateAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_6/p_re_lu_6/alphaAdam/v/dense_6/p_re_lu_6/alphaAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biasAdam/m/dense_7/p_re_lu_7/alphaAdam/v/dense_7/p_re_lu_7/alphaAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biasAdam/m/dense_8/p_re_lu_8/alphaAdam/v/dense_8/p_re_lu_8/alphaAdam/m/dense_9/kernelAdam/v/dense_9/kernelAdam/m/dense_9/biasAdam/v/dense_9/biasAdam/m/dense_9/p_re_lu_9/alphaAdam/v/dense_9/p_re_lu_9/alphaAdam/m/first_layer/kernelAdam/v/first_layer/kernelAdam/m/first_layer/biasAdam/v/first_layer/biasAdam/m/second_layer/kernelAdam/v/second_layer/kernelAdam/m/second_layer/biasAdam/v/second_layer/biasAdam/m/dense_10/kernelAdam/v/dense_10/kernelAdam/m/dense_10/biasAdam/v/dense_10/bias Adam/m/dense_10/p_re_lu_10/alpha Adam/v/dense_10/p_re_lu_10/alphaAdam/m/dense_11/kernelAdam/v/dense_11/kernelAdam/m/dense_11/biasAdam/v/dense_11/bias Adam/m/dense_11/p_re_lu_11/alpha Adam/v/dense_11/p_re_lu_11/alphaAdam/m/thicknesses/kernelAdam/v/thicknesses/kernelAdam/m/thicknesses/biasAdam/v/thicknesses/biastotal_6count_6total_5count_5total_4count_4total_3count_3total_2count_2total_1count_1totalcountConst*e
Tin^
\2Z*
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_2073191
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasfirst_layer/kernelfirst_layer/biassecond_layer/kernelsecond_layer/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasthicknesses/kernelthicknesses/biasdense_6/p_re_lu_6/alphadense_7/p_re_lu_7/alphadense_8/p_re_lu_8/alphadense_9/p_re_lu_9/alphadense_10/p_re_lu_10/alphadense_11/p_re_lu_11/alpha	iterationlearning_rateAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_6/p_re_lu_6/alphaAdam/v/dense_6/p_re_lu_6/alphaAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biasAdam/m/dense_7/p_re_lu_7/alphaAdam/v/dense_7/p_re_lu_7/alphaAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biasAdam/m/dense_8/p_re_lu_8/alphaAdam/v/dense_8/p_re_lu_8/alphaAdam/m/dense_9/kernelAdam/v/dense_9/kernelAdam/m/dense_9/biasAdam/v/dense_9/biasAdam/m/dense_9/p_re_lu_9/alphaAdam/v/dense_9/p_re_lu_9/alphaAdam/m/first_layer/kernelAdam/v/first_layer/kernelAdam/m/first_layer/biasAdam/v/first_layer/biasAdam/m/second_layer/kernelAdam/v/second_layer/kernelAdam/m/second_layer/biasAdam/v/second_layer/biasAdam/m/dense_10/kernelAdam/v/dense_10/kernelAdam/m/dense_10/biasAdam/v/dense_10/bias Adam/m/dense_10/p_re_lu_10/alpha Adam/v/dense_10/p_re_lu_10/alphaAdam/m/dense_11/kernelAdam/v/dense_11/kernelAdam/m/dense_11/biasAdam/v/dense_11/bias Adam/m/dense_11/p_re_lu_11/alpha Adam/v/dense_11/p_re_lu_11/alphaAdam/m/thicknesses/kernelAdam/v/thicknesses/kernelAdam/m/thicknesses/biasAdam/v/thicknesses/biastotal_6count_6total_5count_5total_4count_4total_3count_3total_2count_2total_1count_1totalcount*d
Tin]
[2Y*
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_2073465��
�
�
E__inference_dense_10_layer_call_and_return_conditional_losses_2072421

inputs0
matmul_readvariableop_resource:` -
biasadd_readvariableop_resource: 0
"p_re_lu_10_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�p_re_lu_10/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:` *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
p_re_lu_10/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� x
p_re_lu_10/ReadVariableOpReadVariableOp"p_re_lu_10_readvariableop_resource*
_output_shapes
: *
dtype0]
p_re_lu_10/NegNeg!p_re_lu_10/ReadVariableOp:value:0*
T0*
_output_shapes
: [
p_re_lu_10/Neg_1NegBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
p_re_lu_10/Relu_1Relup_re_lu_10/Neg_1:y:0*
T0*'
_output_shapes
:��������� |
p_re_lu_10/mulMulp_re_lu_10/Neg:y:0p_re_lu_10/Relu_1:activations:0*
T0*'
_output_shapes
:��������� |
p_re_lu_10/addAddV2p_re_lu_10/Relu:activations:0p_re_lu_10/mul:z:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityp_re_lu_10/add:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^p_re_lu_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������`: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp26
p_re_lu_10/ReadVariableOpp_re_lu_10/ReadVariableOp:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
G
+__inference_dropout_7_layer_call_fn_2072320

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_2071115`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
{
+__inference_p_re_lu_7_layer_call_fn_2072550

inputs
unknown:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_p_re_lu_7_layer_call_and_return_conditional_losses_2070682o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
|
+__inference_p_re_lu_6_layer_call_fn_2072531

inputs
unknown:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_p_re_lu_6_layer_call_and_return_conditional_losses_2070661p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�

�
I__inference_second_layer_layer_call_and_return_conditional_losses_2072377

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:��������� `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
*__inference_dense_10_layer_call_fn_2072403

inputs
unknown:` 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_2070986o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������`: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
i
/__inference_concatenate_1_layer_call_fn_2072384
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2070966`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:��������� :��������� :��������� :QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:��������� 
"
_user_specified_name
inputs_0
�
d
F__inference_dropout_7_layer_call_and_return_conditional_losses_2072337

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�E
�	
D__inference_model_1_layer_call_and_return_conditional_losses_2071360

inputs#
dense_6_2071294:
��
dense_6_2071296:	�
dense_6_2071298:	�"
dense_7_2071301:	�@
dense_7_2071303:@
dense_7_2071305:@!
dense_8_2071309:@ 
dense_8_2071311: 
dense_8_2071313: !
dense_9_2071317:  
dense_9_2071319: 
dense_9_2071321: %
first_layer_2071325:  !
first_layer_2071327: &
second_layer_2071330:  "
second_layer_2071332: "
dense_10_2071336:` 
dense_10_2071338: 
dense_10_2071340: "
dense_11_2071344:  
dense_11_2071346: 
dense_11_2071348: %
thicknesses_2071352: !
thicknesses_2071354:
identity

identity_1

identity_2�� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�#first_layer/StatefulPartitionedCall�$second_layer/StatefulPartitionedCall�#thicknesses/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_2071294dense_6_2071296dense_6_2071298*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_2070796�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_2071301dense_7_2071303dense_7_2071305*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_2070822�
dropout_5/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_2071089�
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_8_2071309dense_8_2071311dense_8_2071313*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_2070862�
dropout_6/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_2071102�
dense_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_9_2071317dense_9_2071319dense_9_2071321*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_2070902�
dropout_7/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_2071115�
#first_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0first_layer_2071325first_layer_2071327*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_first_layer_layer_call_and_return_conditional_losses_2070935�
$second_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0second_layer_2071330second_layer_2071332*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_second_layer_layer_call_and_return_conditional_losses_2070952�
concatenate_1/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0,first_layer/StatefulPartitionedCall:output:0-second_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2070966�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_10_2071336dense_10_2071338dense_10_2071340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_2070986�
dropout_8/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_2071139�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_11_2071344dense_11_2071346dense_11_2071348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_2071026�
dropout_9/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_2071152�
#thicknesses/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0thicknesses_2071352thicknesses_2071354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_thicknesses_layer_call_and_return_conditional_losses_2071059{
IdentityIdentity,first_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� ~

Identity_1Identity-second_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� }

Identity_2Identity,thicknesses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall$^first_layer/StatefulPartitionedCall%^second_layer/StatefulPartitionedCall$^thicknesses/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2J
#first_layer/StatefulPartitionedCall#first_layer/StatefulPartitionedCall2L
$second_layer/StatefulPartitionedCall$second_layer/StatefulPartitionedCall2J
#thicknesses/StatefulPartitionedCall#thicknesses/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
G__inference_p_re_lu_11_layer_call_and_return_conditional_losses_2072638

inputs%
readvariableop_resource: 
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
: O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� [
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:��������� V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:��������� W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
d
F__inference_dropout_8_layer_call_and_return_conditional_losses_2072448

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_dense_6_layer_call_and_return_conditional_losses_2070796

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�0
!p_re_lu_6_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�p_re_lu_6/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������[
p_re_lu_6/ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������w
p_re_lu_6/ReadVariableOpReadVariableOp!p_re_lu_6_readvariableop_resource*
_output_shapes	
:�*
dtype0\
p_re_lu_6/NegNeg p_re_lu_6/ReadVariableOp:value:0*
T0*
_output_shapes	
:�[
p_re_lu_6/Neg_1NegBiasAdd:output:0*
T0*(
_output_shapes
:����������`
p_re_lu_6/Relu_1Relup_re_lu_6/Neg_1:y:0*
T0*(
_output_shapes
:����������z
p_re_lu_6/mulMulp_re_lu_6/Neg:y:0p_re_lu_6/Relu_1:activations:0*
T0*(
_output_shapes
:����������z
p_re_lu_6/addAddV2p_re_lu_6/Relu:activations:0p_re_lu_6/mul:z:0*
T0*(
_output_shapes
:����������a
IdentityIdentityp_re_lu_6/add:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^p_re_lu_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:����������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp24
p_re_lu_6/ReadVariableOpp_re_lu_6/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�E
�	
D__inference_model_1_layer_call_and_return_conditional_losses_2071162
input_2#
dense_6_2071071:
��
dense_6_2071073:	�
dense_6_2071075:	�"
dense_7_2071078:	�@
dense_7_2071080:@
dense_7_2071082:@!
dense_8_2071091:@ 
dense_8_2071093: 
dense_8_2071095: !
dense_9_2071104:  
dense_9_2071106: 
dense_9_2071108: %
first_layer_2071117:  !
first_layer_2071119: &
second_layer_2071122:  "
second_layer_2071124: "
dense_10_2071128:` 
dense_10_2071130: 
dense_10_2071132: "
dense_11_2071141:  
dense_11_2071143: 
dense_11_2071145: %
thicknesses_2071154: !
thicknesses_2071156:
identity

identity_1

identity_2�� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�#first_layer/StatefulPartitionedCall�$second_layer/StatefulPartitionedCall�#thicknesses/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_6_2071071dense_6_2071073dense_6_2071075*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_2070796�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_2071078dense_7_2071080dense_7_2071082*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_2070822�
dropout_5/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_2071089�
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_8_2071091dense_8_2071093dense_8_2071095*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_2070862�
dropout_6/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_2071102�
dense_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_9_2071104dense_9_2071106dense_9_2071108*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_2070902�
dropout_7/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_2071115�
#first_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0first_layer_2071117first_layer_2071119*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_first_layer_layer_call_and_return_conditional_losses_2070935�
$second_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0second_layer_2071122second_layer_2071124*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_second_layer_layer_call_and_return_conditional_losses_2070952�
concatenate_1/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0,first_layer/StatefulPartitionedCall:output:0-second_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2070966�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_10_2071128dense_10_2071130dense_10_2071132*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_2070986�
dropout_8/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_2071139�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_11_2071141dense_11_2071143dense_11_2071145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_2071026�
dropout_9/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_2071152�
#thicknesses/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0thicknesses_2071154thicknesses_2071156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_thicknesses_layer_call_and_return_conditional_losses_2071059{
IdentityIdentity,first_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� ~

Identity_1Identity-second_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� }

Identity_2Identity,thicknesses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall$^first_layer/StatefulPartitionedCall%^second_layer/StatefulPartitionedCall$^thicknesses/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2J
#first_layer/StatefulPartitionedCall#first_layer/StatefulPartitionedCall2L
$second_layer/StatefulPartitionedCall$second_layer/StatefulPartitionedCall2J
#thicknesses/StatefulPartitionedCall#thicknesses/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2
�
�
)__inference_model_1_layer_call_fn_2071415
input_2
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�@
	unknown_3:@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:  
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15:` 

unknown_16: 

unknown_17: 

unknown_18:  

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2071360o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2
�

e
F__inference_dropout_9_layer_call_and_return_conditional_losses_2072499

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
G
+__inference_dropout_9_layer_call_fn_2072487

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_2071152`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
{
+__inference_p_re_lu_9_layer_call_fn_2072588

inputs
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_p_re_lu_9_layer_call_and_return_conditional_losses_2070724o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
E__inference_dense_11_layer_call_and_return_conditional_losses_2071026

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 0
"p_re_lu_11_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�p_re_lu_11/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
p_re_lu_11/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� x
p_re_lu_11/ReadVariableOpReadVariableOp"p_re_lu_11_readvariableop_resource*
_output_shapes
: *
dtype0]
p_re_lu_11/NegNeg!p_re_lu_11/ReadVariableOp:value:0*
T0*
_output_shapes
: [
p_re_lu_11/Neg_1NegBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
p_re_lu_11/Relu_1Relup_re_lu_11/Neg_1:y:0*
T0*'
_output_shapes
:��������� |
p_re_lu_11/mulMulp_re_lu_11/Neg:y:0p_re_lu_11/Relu_1:activations:0*
T0*'
_output_shapes
:��������� |
p_re_lu_11/addAddV2p_re_lu_11/Relu:activations:0p_re_lu_11/mul:z:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityp_re_lu_11/add:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^p_re_lu_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp26
p_re_lu_11/ReadVariableOpp_re_lu_11/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_dense_9_layer_call_and_return_conditional_losses_2072310

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: /
!p_re_lu_9_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�p_re_lu_9/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Z
p_re_lu_9/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� v
p_re_lu_9/ReadVariableOpReadVariableOp!p_re_lu_9_readvariableop_resource*
_output_shapes
: *
dtype0[
p_re_lu_9/NegNeg p_re_lu_9/ReadVariableOp:value:0*
T0*
_output_shapes
: Z
p_re_lu_9/Neg_1NegBiasAdd:output:0*
T0*'
_output_shapes
:��������� _
p_re_lu_9/Relu_1Relup_re_lu_9/Neg_1:y:0*
T0*'
_output_shapes
:��������� y
p_re_lu_9/mulMulp_re_lu_9/Neg:y:0p_re_lu_9/Relu_1:activations:0*
T0*'
_output_shapes
:��������� y
p_re_lu_9/addAddV2p_re_lu_9/Relu:activations:0p_re_lu_9/mul:z:0*
T0*'
_output_shapes
:��������� `
IdentityIdentityp_re_lu_9/add:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^p_re_lu_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp24
p_re_lu_9/ReadVariableOpp_re_lu_9/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
d
F__inference_dropout_9_layer_call_and_return_conditional_losses_2072504

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2072392
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:���������`W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:��������� :��������� :��������� :QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:��������� 
"
_user_specified_name
inputs_0
�

�
H__inference_first_layer_layer_call_and_return_conditional_losses_2070935

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:��������� `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
E__inference_dense_10_layer_call_and_return_conditional_losses_2070986

inputs0
matmul_readvariableop_resource:` -
biasadd_readvariableop_resource: 0
"p_re_lu_10_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�p_re_lu_10/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:` *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
p_re_lu_10/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� x
p_re_lu_10/ReadVariableOpReadVariableOp"p_re_lu_10_readvariableop_resource*
_output_shapes
: *
dtype0]
p_re_lu_10/NegNeg!p_re_lu_10/ReadVariableOp:value:0*
T0*
_output_shapes
: [
p_re_lu_10/Neg_1NegBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
p_re_lu_10/Relu_1Relup_re_lu_10/Neg_1:y:0*
T0*'
_output_shapes
:��������� |
p_re_lu_10/mulMulp_re_lu_10/Neg:y:0p_re_lu_10/Relu_1:activations:0*
T0*'
_output_shapes
:��������� |
p_re_lu_10/addAddV2p_re_lu_10/Relu:activations:0p_re_lu_10/mul:z:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityp_re_lu_10/add:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^p_re_lu_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������`: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp26
p_re_lu_10/ReadVariableOpp_re_lu_10/ReadVariableOp:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�	
�
F__inference_p_re_lu_7_layer_call_and_return_conditional_losses_2070682

inputs%
readvariableop_resource:@
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:@O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@[
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
d
F__inference_dropout_7_layer_call_and_return_conditional_losses_2071115

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�M
�
D__inference_model_1_layer_call_and_return_conditional_losses_2071234

inputs#
dense_6_2071168:
��
dense_6_2071170:	�
dense_6_2071172:	�"
dense_7_2071175:	�@
dense_7_2071177:@
dense_7_2071179:@!
dense_8_2071183:@ 
dense_8_2071185: 
dense_8_2071187: !
dense_9_2071191:  
dense_9_2071193: 
dense_9_2071195: %
first_layer_2071199:  !
first_layer_2071201: &
second_layer_2071204:  "
second_layer_2071206: "
dense_10_2071210:` 
dense_10_2071212: 
dense_10_2071214: "
dense_11_2071218:  
dense_11_2071220: 
dense_11_2071222: %
thicknesses_2071226: !
thicknesses_2071228:
identity

identity_1

identity_2�� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�!dropout_6/StatefulPartitionedCall�!dropout_7/StatefulPartitionedCall�!dropout_8/StatefulPartitionedCall�!dropout_9/StatefulPartitionedCall�#first_layer/StatefulPartitionedCall�$second_layer/StatefulPartitionedCall�#thicknesses/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_2071168dense_6_2071170dense_6_2071172*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_2070796�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_2071175dense_7_2071177dense_7_2071179*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_2070822�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_2070842�
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_8_2071183dense_8_2071185dense_8_2071187*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_2070862�
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_2070882�
dense_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_9_2071191dense_9_2071193dense_9_2071195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_2070902�
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_2070922�
#first_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0first_layer_2071199first_layer_2071201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_first_layer_layer_call_and_return_conditional_losses_2070935�
$second_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0second_layer_2071204second_layer_2071206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_second_layer_layer_call_and_return_conditional_losses_2070952�
concatenate_1/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0,first_layer/StatefulPartitionedCall:output:0-second_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2070966�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_10_2071210dense_10_2071212dense_10_2071214*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_2070986�
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_2071006�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_11_2071218dense_11_2071220dense_11_2071222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_2071026�
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_2071046�
#thicknesses/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0thicknesses_2071226thicknesses_2071228*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_thicknesses_layer_call_and_return_conditional_losses_2071059{
IdentityIdentity,first_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� ~

Identity_1Identity-second_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� }

Identity_2Identity,thicknesses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall$^first_layer/StatefulPartitionedCall%^second_layer/StatefulPartitionedCall$^thicknesses/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2J
#first_layer/StatefulPartitionedCall#first_layer/StatefulPartitionedCall2L
$second_layer/StatefulPartitionedCall$second_layer/StatefulPartitionedCall2J
#thicknesses/StatefulPartitionedCall#thicknesses/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

e
F__inference_dropout_9_layer_call_and_return_conditional_losses_2071046

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
*__inference_dense_11_layer_call_fn_2072459

inputs
unknown:  
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_2071026o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_dense_7_layer_call_and_return_conditional_losses_2070822

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@/
!p_re_lu_7_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�p_re_lu_7/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Z
p_re_lu_7/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@v
p_re_lu_7/ReadVariableOpReadVariableOp!p_re_lu_7_readvariableop_resource*
_output_shapes
:@*
dtype0[
p_re_lu_7/NegNeg p_re_lu_7/ReadVariableOp:value:0*
T0*
_output_shapes
:@Z
p_re_lu_7/Neg_1NegBiasAdd:output:0*
T0*'
_output_shapes
:���������@_
p_re_lu_7/Relu_1Relup_re_lu_7/Neg_1:y:0*
T0*'
_output_shapes
:���������@y
p_re_lu_7/mulMulp_re_lu_7/Neg:y:0p_re_lu_7/Relu_1:activations:0*
T0*'
_output_shapes
:���������@y
p_re_lu_7/addAddV2p_re_lu_7/Relu:activations:0p_re_lu_7/mul:z:0*
T0*'
_output_shapes
:���������@`
IdentityIdentityp_re_lu_7/add:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^p_re_lu_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:����������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp24
p_re_lu_7/ReadVariableOpp_re_lu_7/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

e
F__inference_dropout_5_layer_call_and_return_conditional_losses_2070842

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
F__inference_dropout_6_layer_call_and_return_conditional_losses_2072281

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
d
F__inference_dropout_9_layer_call_and_return_conditional_losses_2071152

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
d
F__inference_dropout_8_layer_call_and_return_conditional_losses_2071139

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
H__inference_thicknesses_layer_call_and_return_conditional_losses_2071059

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

e
F__inference_dropout_7_layer_call_and_return_conditional_losses_2070922

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
)__inference_dense_9_layer_call_fn_2072292

inputs
unknown:  
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_2070902o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
H__inference_thicknesses_layer_call_and_return_conditional_losses_2072524

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�
D__inference_model_1_layer_call_and_return_conditional_losses_2072140

inputs:
&dense_6_matmul_readvariableop_resource:
��6
'dense_6_biasadd_readvariableop_resource:	�8
)dense_6_p_re_lu_6_readvariableop_resource:	�9
&dense_7_matmul_readvariableop_resource:	�@5
'dense_7_biasadd_readvariableop_resource:@7
)dense_7_p_re_lu_7_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@ 5
'dense_8_biasadd_readvariableop_resource: 7
)dense_8_p_re_lu_8_readvariableop_resource: 8
&dense_9_matmul_readvariableop_resource:  5
'dense_9_biasadd_readvariableop_resource: 7
)dense_9_p_re_lu_9_readvariableop_resource: <
*first_layer_matmul_readvariableop_resource:  9
+first_layer_biasadd_readvariableop_resource: =
+second_layer_matmul_readvariableop_resource:  :
,second_layer_biasadd_readvariableop_resource: 9
'dense_10_matmul_readvariableop_resource:` 6
(dense_10_biasadd_readvariableop_resource: 9
+dense_10_p_re_lu_10_readvariableop_resource: 9
'dense_11_matmul_readvariableop_resource:  6
(dense_11_biasadd_readvariableop_resource: 9
+dense_11_p_re_lu_11_readvariableop_resource: <
*thicknesses_matmul_readvariableop_resource: 9
+thicknesses_biasadd_readvariableop_resource:
identity

identity_1

identity_2��dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�"dense_10/p_re_lu_10/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�"dense_11/p_re_lu_11/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp� dense_6/p_re_lu_6/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp� dense_7/p_re_lu_7/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp� dense_8/p_re_lu_8/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp� dense_9/p_re_lu_9/ReadVariableOp�"first_layer/BiasAdd/ReadVariableOp�!first_layer/MatMul/ReadVariableOp�#second_layer/BiasAdd/ReadVariableOp�"second_layer/MatMul/ReadVariableOp�"thicknesses/BiasAdd/ReadVariableOp�!thicknesses/MatMul/ReadVariableOp�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0z
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
dense_6/p_re_lu_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 dense_6/p_re_lu_6/ReadVariableOpReadVariableOp)dense_6_p_re_lu_6_readvariableop_resource*
_output_shapes	
:�*
dtype0l
dense_6/p_re_lu_6/NegNeg(dense_6/p_re_lu_6/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
dense_6/p_re_lu_6/Neg_1Negdense_6/BiasAdd:output:0*
T0*(
_output_shapes
:����������p
dense_6/p_re_lu_6/Relu_1Reludense_6/p_re_lu_6/Neg_1:y:0*
T0*(
_output_shapes
:�����������
dense_6/p_re_lu_6/mulMuldense_6/p_re_lu_6/Neg:y:0&dense_6/p_re_lu_6/Relu_1:activations:0*
T0*(
_output_shapes
:�����������
dense_6/p_re_lu_6/addAddV2$dense_6/p_re_lu_6/Relu:activations:0dense_6/p_re_lu_6/mul:z:0*
T0*(
_output_shapes
:�����������
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_7/MatMulMatMuldense_6/p_re_lu_6/add:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@j
dense_7/p_re_lu_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
 dense_7/p_re_lu_7/ReadVariableOpReadVariableOp)dense_7_p_re_lu_7_readvariableop_resource*
_output_shapes
:@*
dtype0k
dense_7/p_re_lu_7/NegNeg(dense_7/p_re_lu_7/ReadVariableOp:value:0*
T0*
_output_shapes
:@j
dense_7/p_re_lu_7/Neg_1Negdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@o
dense_7/p_re_lu_7/Relu_1Reludense_7/p_re_lu_7/Neg_1:y:0*
T0*'
_output_shapes
:���������@�
dense_7/p_re_lu_7/mulMuldense_7/p_re_lu_7/Neg:y:0&dense_7/p_re_lu_7/Relu_1:activations:0*
T0*'
_output_shapes
:���������@�
dense_7/p_re_lu_7/addAddV2$dense_7/p_re_lu_7/Relu:activations:0dense_7/p_re_lu_7/mul:z:0*
T0*'
_output_shapes
:���������@k
dropout_5/IdentityIdentitydense_7/p_re_lu_7/add:z:0*
T0*'
_output_shapes
:���������@�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
dense_8/MatMulMatMuldropout_5/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� j
dense_8/p_re_lu_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
 dense_8/p_re_lu_8/ReadVariableOpReadVariableOp)dense_8_p_re_lu_8_readvariableop_resource*
_output_shapes
: *
dtype0k
dense_8/p_re_lu_8/NegNeg(dense_8/p_re_lu_8/ReadVariableOp:value:0*
T0*
_output_shapes
: j
dense_8/p_re_lu_8/Neg_1Negdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:��������� o
dense_8/p_re_lu_8/Relu_1Reludense_8/p_re_lu_8/Neg_1:y:0*
T0*'
_output_shapes
:��������� �
dense_8/p_re_lu_8/mulMuldense_8/p_re_lu_8/Neg:y:0&dense_8/p_re_lu_8/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
dense_8/p_re_lu_8/addAddV2$dense_8/p_re_lu_8/Relu:activations:0dense_8/p_re_lu_8/mul:z:0*
T0*'
_output_shapes
:��������� k
dropout_6/IdentityIdentitydense_8/p_re_lu_8/add:z:0*
T0*'
_output_shapes
:��������� �
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_9/MatMulMatMuldropout_6/Identity:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� j
dense_9/p_re_lu_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
 dense_9/p_re_lu_9/ReadVariableOpReadVariableOp)dense_9_p_re_lu_9_readvariableop_resource*
_output_shapes
: *
dtype0k
dense_9/p_re_lu_9/NegNeg(dense_9/p_re_lu_9/ReadVariableOp:value:0*
T0*
_output_shapes
: j
dense_9/p_re_lu_9/Neg_1Negdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:��������� o
dense_9/p_re_lu_9/Relu_1Reludense_9/p_re_lu_9/Neg_1:y:0*
T0*'
_output_shapes
:��������� �
dense_9/p_re_lu_9/mulMuldense_9/p_re_lu_9/Neg:y:0&dense_9/p_re_lu_9/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
dense_9/p_re_lu_9/addAddV2$dense_9/p_re_lu_9/Relu:activations:0dense_9/p_re_lu_9/mul:z:0*
T0*'
_output_shapes
:��������� k
dropout_7/IdentityIdentitydense_9/p_re_lu_9/add:z:0*
T0*'
_output_shapes
:��������� �
!first_layer/MatMul/ReadVariableOpReadVariableOp*first_layer_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
first_layer/MatMulMatMuldropout_7/Identity:output:0)first_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
"first_layer/BiasAdd/ReadVariableOpReadVariableOp+first_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
first_layer/BiasAddBiasAddfirst_layer/MatMul:product:0*first_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� n
first_layer/SoftmaxSoftmaxfirst_layer/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
"second_layer/MatMul/ReadVariableOpReadVariableOp+second_layer_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
second_layer/MatMulMatMuldropout_7/Identity:output:0*second_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
#second_layer/BiasAdd/ReadVariableOpReadVariableOp,second_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
second_layer/BiasAddBiasAddsecond_layer/MatMul:product:0+second_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� p
second_layer/SoftmaxSoftmaxsecond_layer/BiasAdd:output:0*
T0*'
_output_shapes
:��������� [
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2dropout_7/Identity:output:0first_layer/Softmax:softmax:0second_layer/Softmax:softmax:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������`�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:` *
dtype0�
dense_10/MatMulMatMulconcatenate_1/concat:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� m
dense_10/p_re_lu_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
"dense_10/p_re_lu_10/ReadVariableOpReadVariableOp+dense_10_p_re_lu_10_readvariableop_resource*
_output_shapes
: *
dtype0o
dense_10/p_re_lu_10/NegNeg*dense_10/p_re_lu_10/ReadVariableOp:value:0*
T0*
_output_shapes
: m
dense_10/p_re_lu_10/Neg_1Negdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:��������� s
dense_10/p_re_lu_10/Relu_1Reludense_10/p_re_lu_10/Neg_1:y:0*
T0*'
_output_shapes
:��������� �
dense_10/p_re_lu_10/mulMuldense_10/p_re_lu_10/Neg:y:0(dense_10/p_re_lu_10/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
dense_10/p_re_lu_10/addAddV2&dense_10/p_re_lu_10/Relu:activations:0dense_10/p_re_lu_10/mul:z:0*
T0*'
_output_shapes
:��������� m
dropout_8/IdentityIdentitydense_10/p_re_lu_10/add:z:0*
T0*'
_output_shapes
:��������� �
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_11/MatMulMatMuldropout_8/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� m
dense_11/p_re_lu_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
"dense_11/p_re_lu_11/ReadVariableOpReadVariableOp+dense_11_p_re_lu_11_readvariableop_resource*
_output_shapes
: *
dtype0o
dense_11/p_re_lu_11/NegNeg*dense_11/p_re_lu_11/ReadVariableOp:value:0*
T0*
_output_shapes
: m
dense_11/p_re_lu_11/Neg_1Negdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:��������� s
dense_11/p_re_lu_11/Relu_1Reludense_11/p_re_lu_11/Neg_1:y:0*
T0*'
_output_shapes
:��������� �
dense_11/p_re_lu_11/mulMuldense_11/p_re_lu_11/Neg:y:0(dense_11/p_re_lu_11/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
dense_11/p_re_lu_11/addAddV2&dense_11/p_re_lu_11/Relu:activations:0dense_11/p_re_lu_11/mul:z:0*
T0*'
_output_shapes
:��������� m
dropout_9/IdentityIdentitydense_11/p_re_lu_11/add:z:0*
T0*'
_output_shapes
:��������� �
!thicknesses/MatMul/ReadVariableOpReadVariableOp*thicknesses_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
thicknesses/MatMulMatMuldropout_9/Identity:output:0)thicknesses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"thicknesses/BiasAdd/ReadVariableOpReadVariableOp+thicknesses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
thicknesses/BiasAddBiasAddthicknesses/MatMul:product:0*thicknesses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
thicknesses/ReluReluthicknesses/BiasAdd:output:0*
T0*'
_output_shapes
:���������l
IdentityIdentityfirst_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:��������� o

Identity_1Identitysecond_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:��������� o

Identity_2Identitythicknesses/Relu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp#^dense_10/p_re_lu_10/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp#^dense_11/p_re_lu_11/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp!^dense_6/p_re_lu_6/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp!^dense_7/p_re_lu_7/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp!^dense_8/p_re_lu_8/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp!^dense_9/p_re_lu_9/ReadVariableOp#^first_layer/BiasAdd/ReadVariableOp"^first_layer/MatMul/ReadVariableOp$^second_layer/BiasAdd/ReadVariableOp#^second_layer/MatMul/ReadVariableOp#^thicknesses/BiasAdd/ReadVariableOp"^thicknesses/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2H
"dense_10/p_re_lu_10/ReadVariableOp"dense_10/p_re_lu_10/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2H
"dense_11/p_re_lu_11/ReadVariableOp"dense_11/p_re_lu_11/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2D
 dense_6/p_re_lu_6/ReadVariableOp dense_6/p_re_lu_6/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2D
 dense_7/p_re_lu_7/ReadVariableOp dense_7/p_re_lu_7/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2D
 dense_8/p_re_lu_8/ReadVariableOp dense_8/p_re_lu_8/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2D
 dense_9/p_re_lu_9/ReadVariableOp dense_9/p_re_lu_9/ReadVariableOp2H
"first_layer/BiasAdd/ReadVariableOp"first_layer/BiasAdd/ReadVariableOp2F
!first_layer/MatMul/ReadVariableOp!first_layer/MatMul/ReadVariableOp2J
#second_layer/BiasAdd/ReadVariableOp#second_layer/BiasAdd/ReadVariableOp2H
"second_layer/MatMul/ReadVariableOp"second_layer/MatMul/ReadVariableOp2H
"thicknesses/BiasAdd/ReadVariableOp"thicknesses/BiasAdd/ReadVariableOp2F
!thicknesses/MatMul/ReadVariableOp!thicknesses/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_thicknesses_layer_call_fn_2072513

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_thicknesses_layer_call_and_return_conditional_losses_2071059o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_dense_8_layer_call_and_return_conditional_losses_2070862

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: /
!p_re_lu_8_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�p_re_lu_8/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Z
p_re_lu_8/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� v
p_re_lu_8/ReadVariableOpReadVariableOp!p_re_lu_8_readvariableop_resource*
_output_shapes
: *
dtype0[
p_re_lu_8/NegNeg p_re_lu_8/ReadVariableOp:value:0*
T0*
_output_shapes
: Z
p_re_lu_8/Neg_1NegBiasAdd:output:0*
T0*'
_output_shapes
:��������� _
p_re_lu_8/Relu_1Relup_re_lu_8/Neg_1:y:0*
T0*'
_output_shapes
:��������� y
p_re_lu_8/mulMulp_re_lu_8/Neg:y:0p_re_lu_8/Relu_1:activations:0*
T0*'
_output_shapes
:��������� y
p_re_lu_8/addAddV2p_re_lu_8/Relu:activations:0p_re_lu_8/mul:z:0*
T0*'
_output_shapes
:��������� `
IdentityIdentityp_re_lu_8/add:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^p_re_lu_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp24
p_re_lu_8/ReadVariableOpp_re_lu_8/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
+__inference_dropout_8_layer_call_fn_2072426

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_2071006o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
)__inference_dense_8_layer_call_fn_2072236

inputs
unknown:@ 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_2070862o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

e
F__inference_dropout_7_layer_call_and_return_conditional_losses_2072332

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
F__inference_p_re_lu_9_layer_call_and_return_conditional_losses_2070724

inputs%
readvariableop_resource: 
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
: O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� [
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:��������� V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:��������� W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�

e
F__inference_dropout_5_layer_call_and_return_conditional_losses_2072220

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

e
F__inference_dropout_6_layer_call_and_return_conditional_losses_2070882

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
)__inference_model_1_layer_call_fn_2071289
input_2
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�@
	unknown_3:@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:  
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15:` 

unknown_16: 

unknown_17: 

unknown_18:  

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2071234o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2
�
d
F__inference_dropout_5_layer_call_and_return_conditional_losses_2071089

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
.__inference_second_layer_layer_call_fn_2072366

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_second_layer_layer_call_and_return_conditional_losses_2070952o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
-__inference_first_layer_layer_call_fn_2072346

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_first_layer_layer_call_and_return_conditional_losses_2070935o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
G
+__inference_dropout_6_layer_call_fn_2072264

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_2071102`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_dense_9_layer_call_and_return_conditional_losses_2070902

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: /
!p_re_lu_9_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�p_re_lu_9/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Z
p_re_lu_9/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� v
p_re_lu_9/ReadVariableOpReadVariableOp!p_re_lu_9_readvariableop_resource*
_output_shapes
: *
dtype0[
p_re_lu_9/NegNeg p_re_lu_9/ReadVariableOp:value:0*
T0*
_output_shapes
: Z
p_re_lu_9/Neg_1NegBiasAdd:output:0*
T0*'
_output_shapes
:��������� _
p_re_lu_9/Relu_1Relup_re_lu_9/Neg_1:y:0*
T0*'
_output_shapes
:��������� y
p_re_lu_9/mulMulp_re_lu_9/Neg:y:0p_re_lu_9/Relu_1:activations:0*
T0*'
_output_shapes
:��������� y
p_re_lu_9/addAddV2p_re_lu_9/Relu:activations:0p_re_lu_9/mul:z:0*
T0*'
_output_shapes
:��������� `
IdentityIdentityp_re_lu_9/add:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^p_re_lu_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp24
p_re_lu_9/ReadVariableOpp_re_lu_9/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_dense_6_layer_call_and_return_conditional_losses_2072169

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�0
!p_re_lu_6_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�p_re_lu_6/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������[
p_re_lu_6/ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������w
p_re_lu_6/ReadVariableOpReadVariableOp!p_re_lu_6_readvariableop_resource*
_output_shapes	
:�*
dtype0\
p_re_lu_6/NegNeg p_re_lu_6/ReadVariableOp:value:0*
T0*
_output_shapes	
:�[
p_re_lu_6/Neg_1NegBiasAdd:output:0*
T0*(
_output_shapes
:����������`
p_re_lu_6/Relu_1Relup_re_lu_6/Neg_1:y:0*
T0*(
_output_shapes
:����������z
p_re_lu_6/mulMulp_re_lu_6/Neg:y:0p_re_lu_6/Relu_1:activations:0*
T0*(
_output_shapes
:����������z
p_re_lu_6/addAddV2p_re_lu_6/Relu:activations:0p_re_lu_6/mul:z:0*
T0*(
_output_shapes
:����������a
IdentityIdentityp_re_lu_6/add:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^p_re_lu_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:����������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp24
p_re_lu_6/ReadVariableOpp_re_lu_6/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
{
+__inference_p_re_lu_8_layer_call_fn_2072569

inputs
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_p_re_lu_8_layer_call_and_return_conditional_losses_2070703o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�	
�
F__inference_p_re_lu_7_layer_call_and_return_conditional_losses_2072562

inputs%
readvariableop_resource:@
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:@O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@[
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
G
+__inference_dropout_8_layer_call_fn_2072431

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_2071139`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
"__inference__wrapped_model_2070648
input_2B
.model_1_dense_6_matmul_readvariableop_resource:
��>
/model_1_dense_6_biasadd_readvariableop_resource:	�@
1model_1_dense_6_p_re_lu_6_readvariableop_resource:	�A
.model_1_dense_7_matmul_readvariableop_resource:	�@=
/model_1_dense_7_biasadd_readvariableop_resource:@?
1model_1_dense_7_p_re_lu_7_readvariableop_resource:@@
.model_1_dense_8_matmul_readvariableop_resource:@ =
/model_1_dense_8_biasadd_readvariableop_resource: ?
1model_1_dense_8_p_re_lu_8_readvariableop_resource: @
.model_1_dense_9_matmul_readvariableop_resource:  =
/model_1_dense_9_biasadd_readvariableop_resource: ?
1model_1_dense_9_p_re_lu_9_readvariableop_resource: D
2model_1_first_layer_matmul_readvariableop_resource:  A
3model_1_first_layer_biasadd_readvariableop_resource: E
3model_1_second_layer_matmul_readvariableop_resource:  B
4model_1_second_layer_biasadd_readvariableop_resource: A
/model_1_dense_10_matmul_readvariableop_resource:` >
0model_1_dense_10_biasadd_readvariableop_resource: A
3model_1_dense_10_p_re_lu_10_readvariableop_resource: A
/model_1_dense_11_matmul_readvariableop_resource:  >
0model_1_dense_11_biasadd_readvariableop_resource: A
3model_1_dense_11_p_re_lu_11_readvariableop_resource: D
2model_1_thicknesses_matmul_readvariableop_resource: A
3model_1_thicknesses_biasadd_readvariableop_resource:
identity

identity_1

identity_2��'model_1/dense_10/BiasAdd/ReadVariableOp�&model_1/dense_10/MatMul/ReadVariableOp�*model_1/dense_10/p_re_lu_10/ReadVariableOp�'model_1/dense_11/BiasAdd/ReadVariableOp�&model_1/dense_11/MatMul/ReadVariableOp�*model_1/dense_11/p_re_lu_11/ReadVariableOp�&model_1/dense_6/BiasAdd/ReadVariableOp�%model_1/dense_6/MatMul/ReadVariableOp�(model_1/dense_6/p_re_lu_6/ReadVariableOp�&model_1/dense_7/BiasAdd/ReadVariableOp�%model_1/dense_7/MatMul/ReadVariableOp�(model_1/dense_7/p_re_lu_7/ReadVariableOp�&model_1/dense_8/BiasAdd/ReadVariableOp�%model_1/dense_8/MatMul/ReadVariableOp�(model_1/dense_8/p_re_lu_8/ReadVariableOp�&model_1/dense_9/BiasAdd/ReadVariableOp�%model_1/dense_9/MatMul/ReadVariableOp�(model_1/dense_9/p_re_lu_9/ReadVariableOp�*model_1/first_layer/BiasAdd/ReadVariableOp�)model_1/first_layer/MatMul/ReadVariableOp�+model_1/second_layer/BiasAdd/ReadVariableOp�*model_1/second_layer/MatMul/ReadVariableOp�*model_1/thicknesses/BiasAdd/ReadVariableOp�)model_1/thicknesses/MatMul/ReadVariableOp�
%model_1/dense_6/MatMul/ReadVariableOpReadVariableOp.model_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model_1/dense_6/MatMulMatMulinput_2-model_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&model_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/dense_6/BiasAddBiasAdd model_1/dense_6/MatMul:product:0.model_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
model_1/dense_6/p_re_lu_6/ReluRelu model_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
(model_1/dense_6/p_re_lu_6/ReadVariableOpReadVariableOp1model_1_dense_6_p_re_lu_6_readvariableop_resource*
_output_shapes	
:�*
dtype0|
model_1/dense_6/p_re_lu_6/NegNeg0model_1/dense_6/p_re_lu_6/ReadVariableOp:value:0*
T0*
_output_shapes	
:�{
model_1/dense_6/p_re_lu_6/Neg_1Neg model_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 model_1/dense_6/p_re_lu_6/Relu_1Relu#model_1/dense_6/p_re_lu_6/Neg_1:y:0*
T0*(
_output_shapes
:�����������
model_1/dense_6/p_re_lu_6/mulMul!model_1/dense_6/p_re_lu_6/Neg:y:0.model_1/dense_6/p_re_lu_6/Relu_1:activations:0*
T0*(
_output_shapes
:�����������
model_1/dense_6/p_re_lu_6/addAddV2,model_1/dense_6/p_re_lu_6/Relu:activations:0!model_1/dense_6/p_re_lu_6/mul:z:0*
T0*(
_output_shapes
:�����������
%model_1/dense_7/MatMul/ReadVariableOpReadVariableOp.model_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model_1/dense_7/MatMulMatMul!model_1/dense_6/p_re_lu_6/add:z:0-model_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&model_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_1/dense_7/BiasAddBiasAdd model_1/dense_7/MatMul:product:0.model_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
model_1/dense_7/p_re_lu_7/ReluRelu model_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
(model_1/dense_7/p_re_lu_7/ReadVariableOpReadVariableOp1model_1_dense_7_p_re_lu_7_readvariableop_resource*
_output_shapes
:@*
dtype0{
model_1/dense_7/p_re_lu_7/NegNeg0model_1/dense_7/p_re_lu_7/ReadVariableOp:value:0*
T0*
_output_shapes
:@z
model_1/dense_7/p_re_lu_7/Neg_1Neg model_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@
 model_1/dense_7/p_re_lu_7/Relu_1Relu#model_1/dense_7/p_re_lu_7/Neg_1:y:0*
T0*'
_output_shapes
:���������@�
model_1/dense_7/p_re_lu_7/mulMul!model_1/dense_7/p_re_lu_7/Neg:y:0.model_1/dense_7/p_re_lu_7/Relu_1:activations:0*
T0*'
_output_shapes
:���������@�
model_1/dense_7/p_re_lu_7/addAddV2,model_1/dense_7/p_re_lu_7/Relu:activations:0!model_1/dense_7/p_re_lu_7/mul:z:0*
T0*'
_output_shapes
:���������@{
model_1/dropout_5/IdentityIdentity!model_1/dense_7/p_re_lu_7/add:z:0*
T0*'
_output_shapes
:���������@�
%model_1/dense_8/MatMul/ReadVariableOpReadVariableOp.model_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
model_1/dense_8/MatMulMatMul#model_1/dropout_5/Identity:output:0-model_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&model_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/dense_8/BiasAddBiasAdd model_1/dense_8/MatMul:product:0.model_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� z
model_1/dense_8/p_re_lu_8/ReluRelu model_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
(model_1/dense_8/p_re_lu_8/ReadVariableOpReadVariableOp1model_1_dense_8_p_re_lu_8_readvariableop_resource*
_output_shapes
: *
dtype0{
model_1/dense_8/p_re_lu_8/NegNeg0model_1/dense_8/p_re_lu_8/ReadVariableOp:value:0*
T0*
_output_shapes
: z
model_1/dense_8/p_re_lu_8/Neg_1Neg model_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 
 model_1/dense_8/p_re_lu_8/Relu_1Relu#model_1/dense_8/p_re_lu_8/Neg_1:y:0*
T0*'
_output_shapes
:��������� �
model_1/dense_8/p_re_lu_8/mulMul!model_1/dense_8/p_re_lu_8/Neg:y:0.model_1/dense_8/p_re_lu_8/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
model_1/dense_8/p_re_lu_8/addAddV2,model_1/dense_8/p_re_lu_8/Relu:activations:0!model_1/dense_8/p_re_lu_8/mul:z:0*
T0*'
_output_shapes
:��������� {
model_1/dropout_6/IdentityIdentity!model_1/dense_8/p_re_lu_8/add:z:0*
T0*'
_output_shapes
:��������� �
%model_1/dense_9/MatMul/ReadVariableOpReadVariableOp.model_1_dense_9_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model_1/dense_9/MatMulMatMul#model_1/dropout_6/Identity:output:0-model_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&model_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/dense_9/BiasAddBiasAdd model_1/dense_9/MatMul:product:0.model_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� z
model_1/dense_9/p_re_lu_9/ReluRelu model_1/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
(model_1/dense_9/p_re_lu_9/ReadVariableOpReadVariableOp1model_1_dense_9_p_re_lu_9_readvariableop_resource*
_output_shapes
: *
dtype0{
model_1/dense_9/p_re_lu_9/NegNeg0model_1/dense_9/p_re_lu_9/ReadVariableOp:value:0*
T0*
_output_shapes
: z
model_1/dense_9/p_re_lu_9/Neg_1Neg model_1/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 
 model_1/dense_9/p_re_lu_9/Relu_1Relu#model_1/dense_9/p_re_lu_9/Neg_1:y:0*
T0*'
_output_shapes
:��������� �
model_1/dense_9/p_re_lu_9/mulMul!model_1/dense_9/p_re_lu_9/Neg:y:0.model_1/dense_9/p_re_lu_9/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
model_1/dense_9/p_re_lu_9/addAddV2,model_1/dense_9/p_re_lu_9/Relu:activations:0!model_1/dense_9/p_re_lu_9/mul:z:0*
T0*'
_output_shapes
:��������� {
model_1/dropout_7/IdentityIdentity!model_1/dense_9/p_re_lu_9/add:z:0*
T0*'
_output_shapes
:��������� �
)model_1/first_layer/MatMul/ReadVariableOpReadVariableOp2model_1_first_layer_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model_1/first_layer/MatMulMatMul#model_1/dropout_7/Identity:output:01model_1/first_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
*model_1/first_layer/BiasAdd/ReadVariableOpReadVariableOp3model_1_first_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/first_layer/BiasAddBiasAdd$model_1/first_layer/MatMul:product:02model_1/first_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� ~
model_1/first_layer/SoftmaxSoftmax$model_1/first_layer/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
*model_1/second_layer/MatMul/ReadVariableOpReadVariableOp3model_1_second_layer_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model_1/second_layer/MatMulMatMul#model_1/dropout_7/Identity:output:02model_1/second_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
+model_1/second_layer/BiasAdd/ReadVariableOpReadVariableOp4model_1_second_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/second_layer/BiasAddBiasAdd%model_1/second_layer/MatMul:product:03model_1/second_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model_1/second_layer/SoftmaxSoftmax%model_1/second_layer/BiasAdd:output:0*
T0*'
_output_shapes
:��������� c
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/concatenate_1/concatConcatV2#model_1/dropout_7/Identity:output:0%model_1/first_layer/Softmax:softmax:0&model_1/second_layer/Softmax:softmax:0*model_1/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������`�
&model_1/dense_10/MatMul/ReadVariableOpReadVariableOp/model_1_dense_10_matmul_readvariableop_resource*
_output_shapes

:` *
dtype0�
model_1/dense_10/MatMulMatMul%model_1/concatenate_1/concat:output:0.model_1/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'model_1/dense_10/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/dense_10/BiasAddBiasAdd!model_1/dense_10/MatMul:product:0/model_1/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� }
 model_1/dense_10/p_re_lu_10/ReluRelu!model_1/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
*model_1/dense_10/p_re_lu_10/ReadVariableOpReadVariableOp3model_1_dense_10_p_re_lu_10_readvariableop_resource*
_output_shapes
: *
dtype0
model_1/dense_10/p_re_lu_10/NegNeg2model_1/dense_10/p_re_lu_10/ReadVariableOp:value:0*
T0*
_output_shapes
: }
!model_1/dense_10/p_re_lu_10/Neg_1Neg!model_1/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
"model_1/dense_10/p_re_lu_10/Relu_1Relu%model_1/dense_10/p_re_lu_10/Neg_1:y:0*
T0*'
_output_shapes
:��������� �
model_1/dense_10/p_re_lu_10/mulMul#model_1/dense_10/p_re_lu_10/Neg:y:00model_1/dense_10/p_re_lu_10/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
model_1/dense_10/p_re_lu_10/addAddV2.model_1/dense_10/p_re_lu_10/Relu:activations:0#model_1/dense_10/p_re_lu_10/mul:z:0*
T0*'
_output_shapes
:��������� }
model_1/dropout_8/IdentityIdentity#model_1/dense_10/p_re_lu_10/add:z:0*
T0*'
_output_shapes
:��������� �
&model_1/dense_11/MatMul/ReadVariableOpReadVariableOp/model_1_dense_11_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model_1/dense_11/MatMulMatMul#model_1/dropout_8/Identity:output:0.model_1/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'model_1/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/dense_11/BiasAddBiasAdd!model_1/dense_11/MatMul:product:0/model_1/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� }
 model_1/dense_11/p_re_lu_11/ReluRelu!model_1/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
*model_1/dense_11/p_re_lu_11/ReadVariableOpReadVariableOp3model_1_dense_11_p_re_lu_11_readvariableop_resource*
_output_shapes
: *
dtype0
model_1/dense_11/p_re_lu_11/NegNeg2model_1/dense_11/p_re_lu_11/ReadVariableOp:value:0*
T0*
_output_shapes
: }
!model_1/dense_11/p_re_lu_11/Neg_1Neg!model_1/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
"model_1/dense_11/p_re_lu_11/Relu_1Relu%model_1/dense_11/p_re_lu_11/Neg_1:y:0*
T0*'
_output_shapes
:��������� �
model_1/dense_11/p_re_lu_11/mulMul#model_1/dense_11/p_re_lu_11/Neg:y:00model_1/dense_11/p_re_lu_11/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
model_1/dense_11/p_re_lu_11/addAddV2.model_1/dense_11/p_re_lu_11/Relu:activations:0#model_1/dense_11/p_re_lu_11/mul:z:0*
T0*'
_output_shapes
:��������� }
model_1/dropout_9/IdentityIdentity#model_1/dense_11/p_re_lu_11/add:z:0*
T0*'
_output_shapes
:��������� �
)model_1/thicknesses/MatMul/ReadVariableOpReadVariableOp2model_1_thicknesses_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
model_1/thicknesses/MatMulMatMul#model_1/dropout_9/Identity:output:01model_1/thicknesses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*model_1/thicknesses/BiasAdd/ReadVariableOpReadVariableOp3model_1_thicknesses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1/thicknesses/BiasAddBiasAdd$model_1/thicknesses/MatMul:product:02model_1/thicknesses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
model_1/thicknesses/ReluRelu$model_1/thicknesses/BiasAdd:output:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%model_1/first_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:��������� w

Identity_1Identity&model_1/second_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:��������� w

Identity_2Identity&model_1/thicknesses/Relu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^model_1/dense_10/BiasAdd/ReadVariableOp'^model_1/dense_10/MatMul/ReadVariableOp+^model_1/dense_10/p_re_lu_10/ReadVariableOp(^model_1/dense_11/BiasAdd/ReadVariableOp'^model_1/dense_11/MatMul/ReadVariableOp+^model_1/dense_11/p_re_lu_11/ReadVariableOp'^model_1/dense_6/BiasAdd/ReadVariableOp&^model_1/dense_6/MatMul/ReadVariableOp)^model_1/dense_6/p_re_lu_6/ReadVariableOp'^model_1/dense_7/BiasAdd/ReadVariableOp&^model_1/dense_7/MatMul/ReadVariableOp)^model_1/dense_7/p_re_lu_7/ReadVariableOp'^model_1/dense_8/BiasAdd/ReadVariableOp&^model_1/dense_8/MatMul/ReadVariableOp)^model_1/dense_8/p_re_lu_8/ReadVariableOp'^model_1/dense_9/BiasAdd/ReadVariableOp&^model_1/dense_9/MatMul/ReadVariableOp)^model_1/dense_9/p_re_lu_9/ReadVariableOp+^model_1/first_layer/BiasAdd/ReadVariableOp*^model_1/first_layer/MatMul/ReadVariableOp,^model_1/second_layer/BiasAdd/ReadVariableOp+^model_1/second_layer/MatMul/ReadVariableOp+^model_1/thicknesses/BiasAdd/ReadVariableOp*^model_1/thicknesses/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2R
'model_1/dense_10/BiasAdd/ReadVariableOp'model_1/dense_10/BiasAdd/ReadVariableOp2P
&model_1/dense_10/MatMul/ReadVariableOp&model_1/dense_10/MatMul/ReadVariableOp2X
*model_1/dense_10/p_re_lu_10/ReadVariableOp*model_1/dense_10/p_re_lu_10/ReadVariableOp2R
'model_1/dense_11/BiasAdd/ReadVariableOp'model_1/dense_11/BiasAdd/ReadVariableOp2P
&model_1/dense_11/MatMul/ReadVariableOp&model_1/dense_11/MatMul/ReadVariableOp2X
*model_1/dense_11/p_re_lu_11/ReadVariableOp*model_1/dense_11/p_re_lu_11/ReadVariableOp2P
&model_1/dense_6/BiasAdd/ReadVariableOp&model_1/dense_6/BiasAdd/ReadVariableOp2N
%model_1/dense_6/MatMul/ReadVariableOp%model_1/dense_6/MatMul/ReadVariableOp2T
(model_1/dense_6/p_re_lu_6/ReadVariableOp(model_1/dense_6/p_re_lu_6/ReadVariableOp2P
&model_1/dense_7/BiasAdd/ReadVariableOp&model_1/dense_7/BiasAdd/ReadVariableOp2N
%model_1/dense_7/MatMul/ReadVariableOp%model_1/dense_7/MatMul/ReadVariableOp2T
(model_1/dense_7/p_re_lu_7/ReadVariableOp(model_1/dense_7/p_re_lu_7/ReadVariableOp2P
&model_1/dense_8/BiasAdd/ReadVariableOp&model_1/dense_8/BiasAdd/ReadVariableOp2N
%model_1/dense_8/MatMul/ReadVariableOp%model_1/dense_8/MatMul/ReadVariableOp2T
(model_1/dense_8/p_re_lu_8/ReadVariableOp(model_1/dense_8/p_re_lu_8/ReadVariableOp2P
&model_1/dense_9/BiasAdd/ReadVariableOp&model_1/dense_9/BiasAdd/ReadVariableOp2N
%model_1/dense_9/MatMul/ReadVariableOp%model_1/dense_9/MatMul/ReadVariableOp2T
(model_1/dense_9/p_re_lu_9/ReadVariableOp(model_1/dense_9/p_re_lu_9/ReadVariableOp2X
*model_1/first_layer/BiasAdd/ReadVariableOp*model_1/first_layer/BiasAdd/ReadVariableOp2V
)model_1/first_layer/MatMul/ReadVariableOp)model_1/first_layer/MatMul/ReadVariableOp2Z
+model_1/second_layer/BiasAdd/ReadVariableOp+model_1/second_layer/BiasAdd/ReadVariableOp2X
*model_1/second_layer/MatMul/ReadVariableOp*model_1/second_layer/MatMul/ReadVariableOp2X
*model_1/thicknesses/BiasAdd/ReadVariableOp*model_1/thicknesses/BiasAdd/ReadVariableOp2V
)model_1/thicknesses/MatMul/ReadVariableOp)model_1/thicknesses/MatMul/ReadVariableOp:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2
�
d
F__inference_dropout_6_layer_call_and_return_conditional_losses_2071102

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
G
+__inference_dropout_5_layer_call_fn_2072208

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_2071089`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
F__inference_p_re_lu_8_layer_call_and_return_conditional_losses_2072581

inputs%
readvariableop_resource: 
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
: O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� [
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:��������� V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:��������� W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�

�
H__inference_first_layer_layer_call_and_return_conditional_losses_2072357

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:��������� `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�M
�
D__inference_model_1_layer_call_and_return_conditional_losses_2071068
input_2#
dense_6_2070797:
��
dense_6_2070799:	�
dense_6_2070801:	�"
dense_7_2070823:	�@
dense_7_2070825:@
dense_7_2070827:@!
dense_8_2070863:@ 
dense_8_2070865: 
dense_8_2070867: !
dense_9_2070903:  
dense_9_2070905: 
dense_9_2070907: %
first_layer_2070936:  !
first_layer_2070938: &
second_layer_2070953:  "
second_layer_2070955: "
dense_10_2070987:` 
dense_10_2070989: 
dense_10_2070991: "
dense_11_2071027:  
dense_11_2071029: 
dense_11_2071031: %
thicknesses_2071060: !
thicknesses_2071062:
identity

identity_1

identity_2�� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�!dropout_6/StatefulPartitionedCall�!dropout_7/StatefulPartitionedCall�!dropout_8/StatefulPartitionedCall�!dropout_9/StatefulPartitionedCall�#first_layer/StatefulPartitionedCall�$second_layer/StatefulPartitionedCall�#thicknesses/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_6_2070797dense_6_2070799dense_6_2070801*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_2070796�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_2070823dense_7_2070825dense_7_2070827*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_2070822�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_2070842�
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_8_2070863dense_8_2070865dense_8_2070867*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_2070862�
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_2070882�
dense_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_9_2070903dense_9_2070905dense_9_2070907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_2070902�
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_2070922�
#first_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0first_layer_2070936first_layer_2070938*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_first_layer_layer_call_and_return_conditional_losses_2070935�
$second_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0second_layer_2070953second_layer_2070955*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_second_layer_layer_call_and_return_conditional_losses_2070952�
concatenate_1/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0,first_layer/StatefulPartitionedCall:output:0-second_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2070966�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_10_2070987dense_10_2070989dense_10_2070991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_2070986�
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_2071006�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_11_2071027dense_11_2071029dense_11_2071031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_2071026�
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_2071046�
#thicknesses/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0thicknesses_2071060thicknesses_2071062*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_thicknesses_layer_call_and_return_conditional_losses_2071059{
IdentityIdentity,first_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� ~

Identity_1Identity-second_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� }

Identity_2Identity,thicknesses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall$^first_layer/StatefulPartitionedCall%^second_layer/StatefulPartitionedCall$^thicknesses/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2J
#first_layer/StatefulPartitionedCall#first_layer/StatefulPartitionedCall2L
$second_layer/StatefulPartitionedCall$second_layer/StatefulPartitionedCall2J
#thicknesses/StatefulPartitionedCall#thicknesses/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2
�

�
I__inference_second_layer_layer_call_and_return_conditional_losses_2070952

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:��������� `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
G__inference_p_re_lu_10_layer_call_and_return_conditional_losses_2070745

inputs%
readvariableop_resource: 
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
: O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� [
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:��������� V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:��������� W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
d
+__inference_dropout_5_layer_call_fn_2072203

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_2070842o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
|
,__inference_p_re_lu_11_layer_call_fn_2072626

inputs
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_p_re_lu_11_layer_call_and_return_conditional_losses_2070766o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
)__inference_model_1_layer_call_fn_2071869

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�@
	unknown_3:@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:  
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15:` 

unknown_16: 

unknown_17: 

unknown_18:  

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2071360o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
G__inference_p_re_lu_10_layer_call_and_return_conditional_losses_2072619

inputs%
readvariableop_resource: 
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
: O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� [
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:��������� V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:��������� W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
d
+__inference_dropout_9_layer_call_fn_2072482

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_2071046o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_dense_7_layer_call_and_return_conditional_losses_2072198

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@/
!p_re_lu_7_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�p_re_lu_7/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Z
p_re_lu_7/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@v
p_re_lu_7/ReadVariableOpReadVariableOp!p_re_lu_7_readvariableop_resource*
_output_shapes
:@*
dtype0[
p_re_lu_7/NegNeg p_re_lu_7/ReadVariableOp:value:0*
T0*
_output_shapes
:@Z
p_re_lu_7/Neg_1NegBiasAdd:output:0*
T0*'
_output_shapes
:���������@_
p_re_lu_7/Relu_1Relup_re_lu_7/Neg_1:y:0*
T0*'
_output_shapes
:���������@y
p_re_lu_7/mulMulp_re_lu_7/Neg:y:0p_re_lu_7/Relu_1:activations:0*
T0*'
_output_shapes
:���������@y
p_re_lu_7/addAddV2p_re_lu_7/Relu:activations:0p_re_lu_7/mul:z:0*
T0*'
_output_shapes
:���������@`
IdentityIdentityp_re_lu_7/add:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^p_re_lu_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:����������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp24
p_re_lu_7/ReadVariableOpp_re_lu_7/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_dense_6_layer_call_fn_2072151

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_2070796p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_p_re_lu_6_layer_call_and_return_conditional_losses_2070661

inputs&
readvariableop_resource:	�
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0H
NegNegReadVariableOp:value:0*
T0*
_output_shapes	
:�O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������\
mulMulNeg:y:0Relu_1:activations:0*
T0*(
_output_shapes
:����������\
addAddV2Relu:activations:0mul:z:0*
T0*(
_output_shapes
:����������W
IdentityIdentityadd:z:0^NoOp*
T0*(
_output_shapes
:����������W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_2071755
input_2
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�@
	unknown_3:@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:  
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15:` 

unknown_16: 

unknown_17: 

unknown_18:  

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2070648o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2
�
�
D__inference_dense_8_layer_call_and_return_conditional_losses_2072254

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: /
!p_re_lu_8_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�p_re_lu_8/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Z
p_re_lu_8/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� v
p_re_lu_8/ReadVariableOpReadVariableOp!p_re_lu_8_readvariableop_resource*
_output_shapes
: *
dtype0[
p_re_lu_8/NegNeg p_re_lu_8/ReadVariableOp:value:0*
T0*
_output_shapes
: Z
p_re_lu_8/Neg_1NegBiasAdd:output:0*
T0*'
_output_shapes
:��������� _
p_re_lu_8/Relu_1Relup_re_lu_8/Neg_1:y:0*
T0*'
_output_shapes
:��������� y
p_re_lu_8/mulMulp_re_lu_8/Neg:y:0p_re_lu_8/Relu_1:activations:0*
T0*'
_output_shapes
:��������� y
p_re_lu_8/addAddV2p_re_lu_8/Relu:activations:0p_re_lu_8/mul:z:0*
T0*'
_output_shapes
:��������� `
IdentityIdentityp_re_lu_8/add:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^p_re_lu_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp24
p_re_lu_8/ReadVariableOpp_re_lu_8/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
í
�
D__inference_model_1_layer_call_and_return_conditional_losses_2072022

inputs:
&dense_6_matmul_readvariableop_resource:
��6
'dense_6_biasadd_readvariableop_resource:	�8
)dense_6_p_re_lu_6_readvariableop_resource:	�9
&dense_7_matmul_readvariableop_resource:	�@5
'dense_7_biasadd_readvariableop_resource:@7
)dense_7_p_re_lu_7_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@ 5
'dense_8_biasadd_readvariableop_resource: 7
)dense_8_p_re_lu_8_readvariableop_resource: 8
&dense_9_matmul_readvariableop_resource:  5
'dense_9_biasadd_readvariableop_resource: 7
)dense_9_p_re_lu_9_readvariableop_resource: <
*first_layer_matmul_readvariableop_resource:  9
+first_layer_biasadd_readvariableop_resource: =
+second_layer_matmul_readvariableop_resource:  :
,second_layer_biasadd_readvariableop_resource: 9
'dense_10_matmul_readvariableop_resource:` 6
(dense_10_biasadd_readvariableop_resource: 9
+dense_10_p_re_lu_10_readvariableop_resource: 9
'dense_11_matmul_readvariableop_resource:  6
(dense_11_biasadd_readvariableop_resource: 9
+dense_11_p_re_lu_11_readvariableop_resource: <
*thicknesses_matmul_readvariableop_resource: 9
+thicknesses_biasadd_readvariableop_resource:
identity

identity_1

identity_2��dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�"dense_10/p_re_lu_10/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�"dense_11/p_re_lu_11/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp� dense_6/p_re_lu_6/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp� dense_7/p_re_lu_7/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp� dense_8/p_re_lu_8/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp� dense_9/p_re_lu_9/ReadVariableOp�"first_layer/BiasAdd/ReadVariableOp�!first_layer/MatMul/ReadVariableOp�#second_layer/BiasAdd/ReadVariableOp�"second_layer/MatMul/ReadVariableOp�"thicknesses/BiasAdd/ReadVariableOp�!thicknesses/MatMul/ReadVariableOp�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0z
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
dense_6/p_re_lu_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 dense_6/p_re_lu_6/ReadVariableOpReadVariableOp)dense_6_p_re_lu_6_readvariableop_resource*
_output_shapes	
:�*
dtype0l
dense_6/p_re_lu_6/NegNeg(dense_6/p_re_lu_6/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
dense_6/p_re_lu_6/Neg_1Negdense_6/BiasAdd:output:0*
T0*(
_output_shapes
:����������p
dense_6/p_re_lu_6/Relu_1Reludense_6/p_re_lu_6/Neg_1:y:0*
T0*(
_output_shapes
:�����������
dense_6/p_re_lu_6/mulMuldense_6/p_re_lu_6/Neg:y:0&dense_6/p_re_lu_6/Relu_1:activations:0*
T0*(
_output_shapes
:�����������
dense_6/p_re_lu_6/addAddV2$dense_6/p_re_lu_6/Relu:activations:0dense_6/p_re_lu_6/mul:z:0*
T0*(
_output_shapes
:�����������
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_7/MatMulMatMuldense_6/p_re_lu_6/add:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@j
dense_7/p_re_lu_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
 dense_7/p_re_lu_7/ReadVariableOpReadVariableOp)dense_7_p_re_lu_7_readvariableop_resource*
_output_shapes
:@*
dtype0k
dense_7/p_re_lu_7/NegNeg(dense_7/p_re_lu_7/ReadVariableOp:value:0*
T0*
_output_shapes
:@j
dense_7/p_re_lu_7/Neg_1Negdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@o
dense_7/p_re_lu_7/Relu_1Reludense_7/p_re_lu_7/Neg_1:y:0*
T0*'
_output_shapes
:���������@�
dense_7/p_re_lu_7/mulMuldense_7/p_re_lu_7/Neg:y:0&dense_7/p_re_lu_7/Relu_1:activations:0*
T0*'
_output_shapes
:���������@�
dense_7/p_re_lu_7/addAddV2$dense_7/p_re_lu_7/Relu:activations:0dense_7/p_re_lu_7/mul:z:0*
T0*'
_output_shapes
:���������@\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_5/dropout/MulMuldense_7/p_re_lu_7/add:z:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:���������@n
dropout_5/dropout/ShapeShapedense_7/p_re_lu_7/add:z:0*
T0*
_output_shapes
::���
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@^
dropout_5/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_5/dropout/SelectV2SelectV2"dropout_5/dropout/GreaterEqual:z:0dropout_5/dropout/Mul:z:0"dropout_5/dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
dense_8/MatMulMatMul#dropout_5/dropout/SelectV2:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� j
dense_8/p_re_lu_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
 dense_8/p_re_lu_8/ReadVariableOpReadVariableOp)dense_8_p_re_lu_8_readvariableop_resource*
_output_shapes
: *
dtype0k
dense_8/p_re_lu_8/NegNeg(dense_8/p_re_lu_8/ReadVariableOp:value:0*
T0*
_output_shapes
: j
dense_8/p_re_lu_8/Neg_1Negdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:��������� o
dense_8/p_re_lu_8/Relu_1Reludense_8/p_re_lu_8/Neg_1:y:0*
T0*'
_output_shapes
:��������� �
dense_8/p_re_lu_8/mulMuldense_8/p_re_lu_8/Neg:y:0&dense_8/p_re_lu_8/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
dense_8/p_re_lu_8/addAddV2$dense_8/p_re_lu_8/Relu:activations:0dense_8/p_re_lu_8/mul:z:0*
T0*'
_output_shapes
:��������� \
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_6/dropout/MulMuldense_8/p_re_lu_8/add:z:0 dropout_6/dropout/Const:output:0*
T0*'
_output_shapes
:��������� n
dropout_6/dropout/ShapeShapedense_8/p_re_lu_8/add:z:0*
T0*
_output_shapes
::���
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� ^
dropout_6/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_6/dropout/SelectV2SelectV2"dropout_6/dropout/GreaterEqual:z:0dropout_6/dropout/Mul:z:0"dropout_6/dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� �
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_9/MatMulMatMul#dropout_6/dropout/SelectV2:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� j
dense_9/p_re_lu_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
 dense_9/p_re_lu_9/ReadVariableOpReadVariableOp)dense_9_p_re_lu_9_readvariableop_resource*
_output_shapes
: *
dtype0k
dense_9/p_re_lu_9/NegNeg(dense_9/p_re_lu_9/ReadVariableOp:value:0*
T0*
_output_shapes
: j
dense_9/p_re_lu_9/Neg_1Negdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:��������� o
dense_9/p_re_lu_9/Relu_1Reludense_9/p_re_lu_9/Neg_1:y:0*
T0*'
_output_shapes
:��������� �
dense_9/p_re_lu_9/mulMuldense_9/p_re_lu_9/Neg:y:0&dense_9/p_re_lu_9/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
dense_9/p_re_lu_9/addAddV2$dense_9/p_re_lu_9/Relu:activations:0dense_9/p_re_lu_9/mul:z:0*
T0*'
_output_shapes
:��������� \
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_7/dropout/MulMuldense_9/p_re_lu_9/add:z:0 dropout_7/dropout/Const:output:0*
T0*'
_output_shapes
:��������� n
dropout_7/dropout/ShapeShapedense_9/p_re_lu_9/add:z:0*
T0*
_output_shapes
::���
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� ^
dropout_7/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_7/dropout/SelectV2SelectV2"dropout_7/dropout/GreaterEqual:z:0dropout_7/dropout/Mul:z:0"dropout_7/dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� �
!first_layer/MatMul/ReadVariableOpReadVariableOp*first_layer_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
first_layer/MatMulMatMul#dropout_7/dropout/SelectV2:output:0)first_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
"first_layer/BiasAdd/ReadVariableOpReadVariableOp+first_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
first_layer/BiasAddBiasAddfirst_layer/MatMul:product:0*first_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� n
first_layer/SoftmaxSoftmaxfirst_layer/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
"second_layer/MatMul/ReadVariableOpReadVariableOp+second_layer_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
second_layer/MatMulMatMul#dropout_7/dropout/SelectV2:output:0*second_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
#second_layer/BiasAdd/ReadVariableOpReadVariableOp,second_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
second_layer/BiasAddBiasAddsecond_layer/MatMul:product:0+second_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� p
second_layer/SoftmaxSoftmaxsecond_layer/BiasAdd:output:0*
T0*'
_output_shapes
:��������� [
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2#dropout_7/dropout/SelectV2:output:0first_layer/Softmax:softmax:0second_layer/Softmax:softmax:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������`�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:` *
dtype0�
dense_10/MatMulMatMulconcatenate_1/concat:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� m
dense_10/p_re_lu_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
"dense_10/p_re_lu_10/ReadVariableOpReadVariableOp+dense_10_p_re_lu_10_readvariableop_resource*
_output_shapes
: *
dtype0o
dense_10/p_re_lu_10/NegNeg*dense_10/p_re_lu_10/ReadVariableOp:value:0*
T0*
_output_shapes
: m
dense_10/p_re_lu_10/Neg_1Negdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:��������� s
dense_10/p_re_lu_10/Relu_1Reludense_10/p_re_lu_10/Neg_1:y:0*
T0*'
_output_shapes
:��������� �
dense_10/p_re_lu_10/mulMuldense_10/p_re_lu_10/Neg:y:0(dense_10/p_re_lu_10/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
dense_10/p_re_lu_10/addAddV2&dense_10/p_re_lu_10/Relu:activations:0dense_10/p_re_lu_10/mul:z:0*
T0*'
_output_shapes
:��������� \
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_8/dropout/MulMuldense_10/p_re_lu_10/add:z:0 dropout_8/dropout/Const:output:0*
T0*'
_output_shapes
:��������� p
dropout_8/dropout/ShapeShapedense_10/p_re_lu_10/add:z:0*
T0*
_output_shapes
::���
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� ^
dropout_8/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_8/dropout/SelectV2SelectV2"dropout_8/dropout/GreaterEqual:z:0dropout_8/dropout/Mul:z:0"dropout_8/dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� �
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_11/MatMulMatMul#dropout_8/dropout/SelectV2:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� m
dense_11/p_re_lu_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
"dense_11/p_re_lu_11/ReadVariableOpReadVariableOp+dense_11_p_re_lu_11_readvariableop_resource*
_output_shapes
: *
dtype0o
dense_11/p_re_lu_11/NegNeg*dense_11/p_re_lu_11/ReadVariableOp:value:0*
T0*
_output_shapes
: m
dense_11/p_re_lu_11/Neg_1Negdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:��������� s
dense_11/p_re_lu_11/Relu_1Reludense_11/p_re_lu_11/Neg_1:y:0*
T0*'
_output_shapes
:��������� �
dense_11/p_re_lu_11/mulMuldense_11/p_re_lu_11/Neg:y:0(dense_11/p_re_lu_11/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
dense_11/p_re_lu_11/addAddV2&dense_11/p_re_lu_11/Relu:activations:0dense_11/p_re_lu_11/mul:z:0*
T0*'
_output_shapes
:��������� \
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_9/dropout/MulMuldense_11/p_re_lu_11/add:z:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:��������� p
dropout_9/dropout/ShapeShapedense_11/p_re_lu_11/add:z:0*
T0*
_output_shapes
::���
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� ^
dropout_9/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_9/dropout/SelectV2SelectV2"dropout_9/dropout/GreaterEqual:z:0dropout_9/dropout/Mul:z:0"dropout_9/dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� �
!thicknesses/MatMul/ReadVariableOpReadVariableOp*thicknesses_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
thicknesses/MatMulMatMul#dropout_9/dropout/SelectV2:output:0)thicknesses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"thicknesses/BiasAdd/ReadVariableOpReadVariableOp+thicknesses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
thicknesses/BiasAddBiasAddthicknesses/MatMul:product:0*thicknesses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
thicknesses/ReluReluthicknesses/BiasAdd:output:0*
T0*'
_output_shapes
:���������l
IdentityIdentityfirst_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:��������� o

Identity_1Identitysecond_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:��������� o

Identity_2Identitythicknesses/Relu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp#^dense_10/p_re_lu_10/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp#^dense_11/p_re_lu_11/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp!^dense_6/p_re_lu_6/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp!^dense_7/p_re_lu_7/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp!^dense_8/p_re_lu_8/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp!^dense_9/p_re_lu_9/ReadVariableOp#^first_layer/BiasAdd/ReadVariableOp"^first_layer/MatMul/ReadVariableOp$^second_layer/BiasAdd/ReadVariableOp#^second_layer/MatMul/ReadVariableOp#^thicknesses/BiasAdd/ReadVariableOp"^thicknesses/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2H
"dense_10/p_re_lu_10/ReadVariableOp"dense_10/p_re_lu_10/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2H
"dense_11/p_re_lu_11/ReadVariableOp"dense_11/p_re_lu_11/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2D
 dense_6/p_re_lu_6/ReadVariableOp dense_6/p_re_lu_6/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2D
 dense_7/p_re_lu_7/ReadVariableOp dense_7/p_re_lu_7/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2D
 dense_8/p_re_lu_8/ReadVariableOp dense_8/p_re_lu_8/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2D
 dense_9/p_re_lu_9/ReadVariableOp dense_9/p_re_lu_9/ReadVariableOp2H
"first_layer/BiasAdd/ReadVariableOp"first_layer/BiasAdd/ReadVariableOp2F
!first_layer/MatMul/ReadVariableOp!first_layer/MatMul/ReadVariableOp2J
#second_layer/BiasAdd/ReadVariableOp#second_layer/BiasAdd/ReadVariableOp2H
"second_layer/MatMul/ReadVariableOp"second_layer/MatMul/ReadVariableOp2H
"thicknesses/BiasAdd/ReadVariableOp"thicknesses/BiasAdd/ReadVariableOp2F
!thicknesses/MatMul/ReadVariableOp!thicknesses/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_p_re_lu_8_layer_call_and_return_conditional_losses_2070703

inputs%
readvariableop_resource: 
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
: O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� [
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:��������� V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:��������� W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
)__inference_dense_7_layer_call_fn_2072180

inputs
unknown:	�@
	unknown_0:@
	unknown_1:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_2070822o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_p_re_lu_9_layer_call_and_return_conditional_losses_2072600

inputs%
readvariableop_resource: 
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
: O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� [
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:��������� V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:��������� W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
d
F__inference_dropout_5_layer_call_and_return_conditional_losses_2072225

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
|
,__inference_p_re_lu_10_layer_call_fn_2072607

inputs
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_p_re_lu_10_layer_call_and_return_conditional_losses_2070745o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
d
+__inference_dropout_7_layer_call_fn_2072315

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_2070922o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
G__inference_p_re_lu_11_layer_call_and_return_conditional_losses_2070766

inputs%
readvariableop_resource: 
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
: O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� [
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:��������� V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:��������� W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
E__inference_dense_11_layer_call_and_return_conditional_losses_2072477

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 0
"p_re_lu_11_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�p_re_lu_11/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
p_re_lu_11/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� x
p_re_lu_11/ReadVariableOpReadVariableOp"p_re_lu_11_readvariableop_resource*
_output_shapes
: *
dtype0]
p_re_lu_11/NegNeg!p_re_lu_11/ReadVariableOp:value:0*
T0*
_output_shapes
: [
p_re_lu_11/Neg_1NegBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
p_re_lu_11/Relu_1Relup_re_lu_11/Neg_1:y:0*
T0*'
_output_shapes
:��������� |
p_re_lu_11/mulMulp_re_lu_11/Neg:y:0p_re_lu_11/Relu_1:activations:0*
T0*'
_output_shapes
:��������� |
p_re_lu_11/addAddV2p_re_lu_11/Relu:activations:0p_re_lu_11/mul:z:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityp_re_lu_11/add:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^p_re_lu_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp26
p_re_lu_11/ReadVariableOpp_re_lu_11/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
F__inference_p_re_lu_6_layer_call_and_return_conditional_losses_2072543

inputs&
readvariableop_resource:	�
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0H
NegNegReadVariableOp:value:0*
T0*
_output_shapes	
:�O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������\
mulMulNeg:y:0Relu_1:activations:0*
T0*(
_output_shapes
:����������\
addAddV2Relu:activations:0mul:z:0*
T0*(
_output_shapes
:����������W
IdentityIdentityadd:z:0^NoOp*
T0*(
_output_shapes
:����������W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�

e
F__inference_dropout_8_layer_call_and_return_conditional_losses_2071006

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

e
F__inference_dropout_8_layer_call_and_return_conditional_losses_2072443

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
d
+__inference_dropout_6_layer_call_fn_2072259

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_2070882o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�5
#__inference__traced_restore_2073465
file_prefix3
assignvariableop_dense_6_kernel:
��.
assignvariableop_1_dense_6_bias:	�4
!assignvariableop_2_dense_7_kernel:	�@-
assignvariableop_3_dense_7_bias:@3
!assignvariableop_4_dense_8_kernel:@ -
assignvariableop_5_dense_8_bias: 3
!assignvariableop_6_dense_9_kernel:  -
assignvariableop_7_dense_9_bias: 7
%assignvariableop_8_first_layer_kernel:  1
#assignvariableop_9_first_layer_bias: 9
'assignvariableop_10_second_layer_kernel:  3
%assignvariableop_11_second_layer_bias: 5
#assignvariableop_12_dense_10_kernel:` /
!assignvariableop_13_dense_10_bias: 5
#assignvariableop_14_dense_11_kernel:  /
!assignvariableop_15_dense_11_bias: 8
&assignvariableop_16_thicknesses_kernel: 2
$assignvariableop_17_thicknesses_bias::
+assignvariableop_18_dense_6_p_re_lu_6_alpha:	�9
+assignvariableop_19_dense_7_p_re_lu_7_alpha:@9
+assignvariableop_20_dense_8_p_re_lu_8_alpha: 9
+assignvariableop_21_dense_9_p_re_lu_9_alpha: ;
-assignvariableop_22_dense_10_p_re_lu_10_alpha: ;
-assignvariableop_23_dense_11_p_re_lu_11_alpha: '
assignvariableop_24_iteration:	 +
!assignvariableop_25_learning_rate: =
)assignvariableop_26_adam_m_dense_6_kernel:
��=
)assignvariableop_27_adam_v_dense_6_kernel:
��6
'assignvariableop_28_adam_m_dense_6_bias:	�6
'assignvariableop_29_adam_v_dense_6_bias:	�A
2assignvariableop_30_adam_m_dense_6_p_re_lu_6_alpha:	�A
2assignvariableop_31_adam_v_dense_6_p_re_lu_6_alpha:	�<
)assignvariableop_32_adam_m_dense_7_kernel:	�@<
)assignvariableop_33_adam_v_dense_7_kernel:	�@5
'assignvariableop_34_adam_m_dense_7_bias:@5
'assignvariableop_35_adam_v_dense_7_bias:@@
2assignvariableop_36_adam_m_dense_7_p_re_lu_7_alpha:@@
2assignvariableop_37_adam_v_dense_7_p_re_lu_7_alpha:@;
)assignvariableop_38_adam_m_dense_8_kernel:@ ;
)assignvariableop_39_adam_v_dense_8_kernel:@ 5
'assignvariableop_40_adam_m_dense_8_bias: 5
'assignvariableop_41_adam_v_dense_8_bias: @
2assignvariableop_42_adam_m_dense_8_p_re_lu_8_alpha: @
2assignvariableop_43_adam_v_dense_8_p_re_lu_8_alpha: ;
)assignvariableop_44_adam_m_dense_9_kernel:  ;
)assignvariableop_45_adam_v_dense_9_kernel:  5
'assignvariableop_46_adam_m_dense_9_bias: 5
'assignvariableop_47_adam_v_dense_9_bias: @
2assignvariableop_48_adam_m_dense_9_p_re_lu_9_alpha: @
2assignvariableop_49_adam_v_dense_9_p_re_lu_9_alpha: ?
-assignvariableop_50_adam_m_first_layer_kernel:  ?
-assignvariableop_51_adam_v_first_layer_kernel:  9
+assignvariableop_52_adam_m_first_layer_bias: 9
+assignvariableop_53_adam_v_first_layer_bias: @
.assignvariableop_54_adam_m_second_layer_kernel:  @
.assignvariableop_55_adam_v_second_layer_kernel:  :
,assignvariableop_56_adam_m_second_layer_bias: :
,assignvariableop_57_adam_v_second_layer_bias: <
*assignvariableop_58_adam_m_dense_10_kernel:` <
*assignvariableop_59_adam_v_dense_10_kernel:` 6
(assignvariableop_60_adam_m_dense_10_bias: 6
(assignvariableop_61_adam_v_dense_10_bias: B
4assignvariableop_62_adam_m_dense_10_p_re_lu_10_alpha: B
4assignvariableop_63_adam_v_dense_10_p_re_lu_10_alpha: <
*assignvariableop_64_adam_m_dense_11_kernel:  <
*assignvariableop_65_adam_v_dense_11_kernel:  6
(assignvariableop_66_adam_m_dense_11_bias: 6
(assignvariableop_67_adam_v_dense_11_bias: B
4assignvariableop_68_adam_m_dense_11_p_re_lu_11_alpha: B
4assignvariableop_69_adam_v_dense_11_p_re_lu_11_alpha: ?
-assignvariableop_70_adam_m_thicknesses_kernel: ?
-assignvariableop_71_adam_v_thicknesses_kernel: 9
+assignvariableop_72_adam_m_thicknesses_bias:9
+assignvariableop_73_adam_v_thicknesses_bias:%
assignvariableop_74_total_6: %
assignvariableop_75_count_6: %
assignvariableop_76_total_5: %
assignvariableop_77_count_5: %
assignvariableop_78_total_4: %
assignvariableop_79_count_4: %
assignvariableop_80_total_3: %
assignvariableop_81_count_3: %
assignvariableop_82_total_2: %
assignvariableop_83_count_2: %
assignvariableop_84_total_1: %
assignvariableop_85_count_1: #
assignvariableop_86_total: #
assignvariableop_87_count: 
identity_89��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_9�$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*�$
value�$B�$YB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*�
value�B�YB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*g
dtypes]
[2Y	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_7_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_7_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_8_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_8_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_9_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_9_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_first_layer_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_first_layer_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_second_layer_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp%assignvariableop_11_second_layer_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_10_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_10_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_11_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_11_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp&assignvariableop_16_thicknesses_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp$assignvariableop_17_thicknesses_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp+assignvariableop_18_dense_6_p_re_lu_6_alphaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp+assignvariableop_19_dense_7_p_re_lu_7_alphaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_dense_8_p_re_lu_8_alphaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp+assignvariableop_21_dense_9_p_re_lu_9_alphaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp-assignvariableop_22_dense_10_p_re_lu_10_alphaIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp-assignvariableop_23_dense_11_p_re_lu_11_alphaIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_iterationIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_learning_rateIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_m_dense_6_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_v_dense_6_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_m_dense_6_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_v_dense_6_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_m_dense_6_p_re_lu_6_alphaIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_v_dense_6_p_re_lu_6_alphaIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_m_dense_7_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_v_dense_7_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_m_dense_7_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_v_dense_7_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp2assignvariableop_36_adam_m_dense_7_p_re_lu_7_alphaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp2assignvariableop_37_adam_v_dense_7_p_re_lu_7_alphaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_m_dense_8_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_v_dense_8_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_m_dense_8_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_v_dense_8_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp2assignvariableop_42_adam_m_dense_8_p_re_lu_8_alphaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp2assignvariableop_43_adam_v_dense_8_p_re_lu_8_alphaIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_m_dense_9_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_v_dense_9_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_m_dense_9_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_v_dense_9_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp2assignvariableop_48_adam_m_dense_9_p_re_lu_9_alphaIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp2assignvariableop_49_adam_v_dense_9_p_re_lu_9_alphaIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp-assignvariableop_50_adam_m_first_layer_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp-assignvariableop_51_adam_v_first_layer_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp+assignvariableop_52_adam_m_first_layer_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_v_first_layer_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp.assignvariableop_54_adam_m_second_layer_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp.assignvariableop_55_adam_v_second_layer_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp,assignvariableop_56_adam_m_second_layer_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_v_second_layer_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_m_dense_10_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_v_dense_10_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_m_dense_10_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_v_dense_10_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp4assignvariableop_62_adam_m_dense_10_p_re_lu_10_alphaIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp4assignvariableop_63_adam_v_dense_10_p_re_lu_10_alphaIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_m_dense_11_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_v_dense_11_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_m_dense_11_biasIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_v_dense_11_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp4assignvariableop_68_adam_m_dense_11_p_re_lu_11_alphaIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp4assignvariableop_69_adam_v_dense_11_p_re_lu_11_alphaIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp-assignvariableop_70_adam_m_thicknesses_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp-assignvariableop_71_adam_v_thicknesses_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_m_thicknesses_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_v_thicknesses_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOpassignvariableop_74_total_6Identity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOpassignvariableop_75_count_6Identity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOpassignvariableop_76_total_5Identity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOpassignvariableop_77_count_5Identity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOpassignvariableop_78_total_4Identity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOpassignvariableop_79_count_4Identity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOpassignvariableop_80_total_3Identity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOpassignvariableop_81_count_3Identity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOpassignvariableop_82_total_2Identity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOpassignvariableop_83_count_2Identity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOpassignvariableop_84_total_1Identity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOpassignvariableop_85_count_1Identity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOpassignvariableop_86_totalIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOpassignvariableop_87_countIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_88Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_89IdentityIdentity_88:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_89Identity_89:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2070966

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:���������`W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:��������� :��������� :��������� :OK
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
)__inference_model_1_layer_call_fn_2071812

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�@
	unknown_3:@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:  
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15:` 

unknown_16: 

unknown_17: 

unknown_18:  

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2071234o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�O
 __inference__traced_save_2073191
file_prefix9
%read_disablecopyonread_dense_6_kernel:
��4
%read_1_disablecopyonread_dense_6_bias:	�:
'read_2_disablecopyonread_dense_7_kernel:	�@3
%read_3_disablecopyonread_dense_7_bias:@9
'read_4_disablecopyonread_dense_8_kernel:@ 3
%read_5_disablecopyonread_dense_8_bias: 9
'read_6_disablecopyonread_dense_9_kernel:  3
%read_7_disablecopyonread_dense_9_bias: =
+read_8_disablecopyonread_first_layer_kernel:  7
)read_9_disablecopyonread_first_layer_bias: ?
-read_10_disablecopyonread_second_layer_kernel:  9
+read_11_disablecopyonread_second_layer_bias: ;
)read_12_disablecopyonread_dense_10_kernel:` 5
'read_13_disablecopyonread_dense_10_bias: ;
)read_14_disablecopyonread_dense_11_kernel:  5
'read_15_disablecopyonread_dense_11_bias: >
,read_16_disablecopyonread_thicknesses_kernel: 8
*read_17_disablecopyonread_thicknesses_bias:@
1read_18_disablecopyonread_dense_6_p_re_lu_6_alpha:	�?
1read_19_disablecopyonread_dense_7_p_re_lu_7_alpha:@?
1read_20_disablecopyonread_dense_8_p_re_lu_8_alpha: ?
1read_21_disablecopyonread_dense_9_p_re_lu_9_alpha: A
3read_22_disablecopyonread_dense_10_p_re_lu_10_alpha: A
3read_23_disablecopyonread_dense_11_p_re_lu_11_alpha: -
#read_24_disablecopyonread_iteration:	 1
'read_25_disablecopyonread_learning_rate: C
/read_26_disablecopyonread_adam_m_dense_6_kernel:
��C
/read_27_disablecopyonread_adam_v_dense_6_kernel:
��<
-read_28_disablecopyonread_adam_m_dense_6_bias:	�<
-read_29_disablecopyonread_adam_v_dense_6_bias:	�G
8read_30_disablecopyonread_adam_m_dense_6_p_re_lu_6_alpha:	�G
8read_31_disablecopyonread_adam_v_dense_6_p_re_lu_6_alpha:	�B
/read_32_disablecopyonread_adam_m_dense_7_kernel:	�@B
/read_33_disablecopyonread_adam_v_dense_7_kernel:	�@;
-read_34_disablecopyonread_adam_m_dense_7_bias:@;
-read_35_disablecopyonread_adam_v_dense_7_bias:@F
8read_36_disablecopyonread_adam_m_dense_7_p_re_lu_7_alpha:@F
8read_37_disablecopyonread_adam_v_dense_7_p_re_lu_7_alpha:@A
/read_38_disablecopyonread_adam_m_dense_8_kernel:@ A
/read_39_disablecopyonread_adam_v_dense_8_kernel:@ ;
-read_40_disablecopyonread_adam_m_dense_8_bias: ;
-read_41_disablecopyonread_adam_v_dense_8_bias: F
8read_42_disablecopyonread_adam_m_dense_8_p_re_lu_8_alpha: F
8read_43_disablecopyonread_adam_v_dense_8_p_re_lu_8_alpha: A
/read_44_disablecopyonread_adam_m_dense_9_kernel:  A
/read_45_disablecopyonread_adam_v_dense_9_kernel:  ;
-read_46_disablecopyonread_adam_m_dense_9_bias: ;
-read_47_disablecopyonread_adam_v_dense_9_bias: F
8read_48_disablecopyonread_adam_m_dense_9_p_re_lu_9_alpha: F
8read_49_disablecopyonread_adam_v_dense_9_p_re_lu_9_alpha: E
3read_50_disablecopyonread_adam_m_first_layer_kernel:  E
3read_51_disablecopyonread_adam_v_first_layer_kernel:  ?
1read_52_disablecopyonread_adam_m_first_layer_bias: ?
1read_53_disablecopyonread_adam_v_first_layer_bias: F
4read_54_disablecopyonread_adam_m_second_layer_kernel:  F
4read_55_disablecopyonread_adam_v_second_layer_kernel:  @
2read_56_disablecopyonread_adam_m_second_layer_bias: @
2read_57_disablecopyonread_adam_v_second_layer_bias: B
0read_58_disablecopyonread_adam_m_dense_10_kernel:` B
0read_59_disablecopyonread_adam_v_dense_10_kernel:` <
.read_60_disablecopyonread_adam_m_dense_10_bias: <
.read_61_disablecopyonread_adam_v_dense_10_bias: H
:read_62_disablecopyonread_adam_m_dense_10_p_re_lu_10_alpha: H
:read_63_disablecopyonread_adam_v_dense_10_p_re_lu_10_alpha: B
0read_64_disablecopyonread_adam_m_dense_11_kernel:  B
0read_65_disablecopyonread_adam_v_dense_11_kernel:  <
.read_66_disablecopyonread_adam_m_dense_11_bias: <
.read_67_disablecopyonread_adam_v_dense_11_bias: H
:read_68_disablecopyonread_adam_m_dense_11_p_re_lu_11_alpha: H
:read_69_disablecopyonread_adam_v_dense_11_p_re_lu_11_alpha: E
3read_70_disablecopyonread_adam_m_thicknesses_kernel: E
3read_71_disablecopyonread_adam_v_thicknesses_kernel: ?
1read_72_disablecopyonread_adam_m_thicknesses_bias:?
1read_73_disablecopyonread_adam_v_thicknesses_bias:+
!read_74_disablecopyonread_total_6: +
!read_75_disablecopyonread_count_6: +
!read_76_disablecopyonread_total_5: +
!read_77_disablecopyonread_count_5: +
!read_78_disablecopyonread_total_4: +
!read_79_disablecopyonread_count_4: +
!read_80_disablecopyonread_total_3: +
!read_81_disablecopyonread_count_3: +
!read_82_disablecopyonread_total_2: +
!read_83_disablecopyonread_count_2: +
!read_84_disablecopyonread_total_1: +
!read_85_disablecopyonread_count_1: )
read_86_disablecopyonread_total: )
read_87_disablecopyonread_count: 
savev2_const
identity_177��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: w
Read/DisableCopyOnReadDisableCopyOnRead%read_disablecopyonread_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp%read_disablecopyonread_dense_6_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_dense_6_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_7_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_7_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_8_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:@ y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_8_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_dense_9_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:  y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_dense_9_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_8/DisableCopyOnReadDisableCopyOnRead+read_8_disablecopyonread_first_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp+read_8_disablecopyonread_first_layer_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:  }
Read_9/DisableCopyOnReadDisableCopyOnRead)read_9_disablecopyonread_first_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp)read_9_disablecopyonread_first_layer_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead-read_10_disablecopyonread_second_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp-read_10_disablecopyonread_second_layer_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_11/DisableCopyOnReadDisableCopyOnRead+read_11_disablecopyonread_second_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp+read_11_disablecopyonread_second_layer_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_dense_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_dense_10_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:` *
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:` e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:` |
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_dense_10_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_dense_10_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_14/DisableCopyOnReadDisableCopyOnRead)read_14_disablecopyonread_dense_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp)read_14_disablecopyonread_dense_11_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:  |
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_dense_11_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_dense_11_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_16/DisableCopyOnReadDisableCopyOnRead,read_16_disablecopyonread_thicknesses_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp,read_16_disablecopyonread_thicknesses_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

: 
Read_17/DisableCopyOnReadDisableCopyOnRead*read_17_disablecopyonread_thicknesses_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp*read_17_disablecopyonread_thicknesses_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnRead1read_18_disablecopyonread_dense_6_p_re_lu_6_alpha"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp1read_18_disablecopyonread_dense_6_p_re_lu_6_alpha^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_19/DisableCopyOnReadDisableCopyOnRead1read_19_disablecopyonread_dense_7_p_re_lu_7_alpha"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp1read_19_disablecopyonread_dense_7_p_re_lu_7_alpha^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_20/DisableCopyOnReadDisableCopyOnRead1read_20_disablecopyonread_dense_8_p_re_lu_8_alpha"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp1read_20_disablecopyonread_dense_8_p_re_lu_8_alpha^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_21/DisableCopyOnReadDisableCopyOnRead1read_21_disablecopyonread_dense_9_p_re_lu_9_alpha"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp1read_21_disablecopyonread_dense_9_p_re_lu_9_alpha^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_22/DisableCopyOnReadDisableCopyOnRead3read_22_disablecopyonread_dense_10_p_re_lu_10_alpha"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp3read_22_disablecopyonread_dense_10_p_re_lu_10_alpha^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_23/DisableCopyOnReadDisableCopyOnRead3read_23_disablecopyonread_dense_11_p_re_lu_11_alpha"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp3read_23_disablecopyonread_dense_11_p_re_lu_11_alpha^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_24/DisableCopyOnReadDisableCopyOnRead#read_24_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp#read_24_disablecopyonread_iteration^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_25/DisableCopyOnReadDisableCopyOnRead'read_25_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp'read_25_disablecopyonread_learning_rate^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_26/DisableCopyOnReadDisableCopyOnRead/read_26_disablecopyonread_adam_m_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp/read_26_disablecopyonread_adam_m_dense_6_kernel^Read_26/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_27/DisableCopyOnReadDisableCopyOnRead/read_27_disablecopyonread_adam_v_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp/read_27_disablecopyonread_adam_v_dense_6_kernel^Read_27/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_28/DisableCopyOnReadDisableCopyOnRead-read_28_disablecopyonread_adam_m_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp-read_28_disablecopyonread_adam_m_dense_6_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_29/DisableCopyOnReadDisableCopyOnRead-read_29_disablecopyonread_adam_v_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp-read_29_disablecopyonread_adam_v_dense_6_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_30/DisableCopyOnReadDisableCopyOnRead8read_30_disablecopyonread_adam_m_dense_6_p_re_lu_6_alpha"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp8read_30_disablecopyonread_adam_m_dense_6_p_re_lu_6_alpha^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_31/DisableCopyOnReadDisableCopyOnRead8read_31_disablecopyonread_adam_v_dense_6_p_re_lu_6_alpha"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp8read_31_disablecopyonread_adam_v_dense_6_p_re_lu_6_alpha^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_32/DisableCopyOnReadDisableCopyOnRead/read_32_disablecopyonread_adam_m_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp/read_32_disablecopyonread_adam_m_dense_7_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_33/DisableCopyOnReadDisableCopyOnRead/read_33_disablecopyonread_adam_v_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp/read_33_disablecopyonread_adam_v_dense_7_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_34/DisableCopyOnReadDisableCopyOnRead-read_34_disablecopyonread_adam_m_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp-read_34_disablecopyonread_adam_m_dense_7_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_35/DisableCopyOnReadDisableCopyOnRead-read_35_disablecopyonread_adam_v_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp-read_35_disablecopyonread_adam_v_dense_7_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_36/DisableCopyOnReadDisableCopyOnRead8read_36_disablecopyonread_adam_m_dense_7_p_re_lu_7_alpha"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp8read_36_disablecopyonread_adam_m_dense_7_p_re_lu_7_alpha^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_37/DisableCopyOnReadDisableCopyOnRead8read_37_disablecopyonread_adam_v_dense_7_p_re_lu_7_alpha"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp8read_37_disablecopyonread_adam_v_dense_7_p_re_lu_7_alpha^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_38/DisableCopyOnReadDisableCopyOnRead/read_38_disablecopyonread_adam_m_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp/read_38_disablecopyonread_adam_m_dense_8_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes

:@ �
Read_39/DisableCopyOnReadDisableCopyOnRead/read_39_disablecopyonread_adam_v_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp/read_39_disablecopyonread_adam_v_dense_8_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes

:@ �
Read_40/DisableCopyOnReadDisableCopyOnRead-read_40_disablecopyonread_adam_m_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp-read_40_disablecopyonread_adam_m_dense_8_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_41/DisableCopyOnReadDisableCopyOnRead-read_41_disablecopyonread_adam_v_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp-read_41_disablecopyonread_adam_v_dense_8_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_42/DisableCopyOnReadDisableCopyOnRead8read_42_disablecopyonread_adam_m_dense_8_p_re_lu_8_alpha"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp8read_42_disablecopyonread_adam_m_dense_8_p_re_lu_8_alpha^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_43/DisableCopyOnReadDisableCopyOnRead8read_43_disablecopyonread_adam_v_dense_8_p_re_lu_8_alpha"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp8read_43_disablecopyonread_adam_v_dense_8_p_re_lu_8_alpha^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_44/DisableCopyOnReadDisableCopyOnRead/read_44_disablecopyonread_adam_m_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp/read_44_disablecopyonread_adam_m_dense_9_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_45/DisableCopyOnReadDisableCopyOnRead/read_45_disablecopyonread_adam_v_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp/read_45_disablecopyonread_adam_v_dense_9_kernel^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_46/DisableCopyOnReadDisableCopyOnRead-read_46_disablecopyonread_adam_m_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp-read_46_disablecopyonread_adam_m_dense_9_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_47/DisableCopyOnReadDisableCopyOnRead-read_47_disablecopyonread_adam_v_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp-read_47_disablecopyonread_adam_v_dense_9_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_48/DisableCopyOnReadDisableCopyOnRead8read_48_disablecopyonread_adam_m_dense_9_p_re_lu_9_alpha"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp8read_48_disablecopyonread_adam_m_dense_9_p_re_lu_9_alpha^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_49/DisableCopyOnReadDisableCopyOnRead8read_49_disablecopyonread_adam_v_dense_9_p_re_lu_9_alpha"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp8read_49_disablecopyonread_adam_v_dense_9_p_re_lu_9_alpha^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_50/DisableCopyOnReadDisableCopyOnRead3read_50_disablecopyonread_adam_m_first_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp3read_50_disablecopyonread_adam_m_first_layer_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0p
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  g
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_51/DisableCopyOnReadDisableCopyOnRead3read_51_disablecopyonread_adam_v_first_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp3read_51_disablecopyonread_adam_v_first_layer_kernel^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0p
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  g
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_52/DisableCopyOnReadDisableCopyOnRead1read_52_disablecopyonread_adam_m_first_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp1read_52_disablecopyonread_adam_m_first_layer_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_53/DisableCopyOnReadDisableCopyOnRead1read_53_disablecopyonread_adam_v_first_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp1read_53_disablecopyonread_adam_v_first_layer_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_54/DisableCopyOnReadDisableCopyOnRead4read_54_disablecopyonread_adam_m_second_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp4read_54_disablecopyonread_adam_m_second_layer_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0p
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  g
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_55/DisableCopyOnReadDisableCopyOnRead4read_55_disablecopyonread_adam_v_second_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp4read_55_disablecopyonread_adam_v_second_layer_kernel^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0p
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  g
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_56/DisableCopyOnReadDisableCopyOnRead2read_56_disablecopyonread_adam_m_second_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp2read_56_disablecopyonread_adam_m_second_layer_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_57/DisableCopyOnReadDisableCopyOnRead2read_57_disablecopyonread_adam_v_second_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp2read_57_disablecopyonread_adam_v_second_layer_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_58/DisableCopyOnReadDisableCopyOnRead0read_58_disablecopyonread_adam_m_dense_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp0read_58_disablecopyonread_adam_m_dense_10_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:` *
dtype0p
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:` g
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes

:` �
Read_59/DisableCopyOnReadDisableCopyOnRead0read_59_disablecopyonread_adam_v_dense_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp0read_59_disablecopyonread_adam_v_dense_10_kernel^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:` *
dtype0p
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:` g
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes

:` �
Read_60/DisableCopyOnReadDisableCopyOnRead.read_60_disablecopyonread_adam_m_dense_10_bias"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp.read_60_disablecopyonread_adam_m_dense_10_bias^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_61/DisableCopyOnReadDisableCopyOnRead.read_61_disablecopyonread_adam_v_dense_10_bias"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp.read_61_disablecopyonread_adam_v_dense_10_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_62/DisableCopyOnReadDisableCopyOnRead:read_62_disablecopyonread_adam_m_dense_10_p_re_lu_10_alpha"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp:read_62_disablecopyonread_adam_m_dense_10_p_re_lu_10_alpha^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_63/DisableCopyOnReadDisableCopyOnRead:read_63_disablecopyonread_adam_v_dense_10_p_re_lu_10_alpha"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp:read_63_disablecopyonread_adam_v_dense_10_p_re_lu_10_alpha^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_64/DisableCopyOnReadDisableCopyOnRead0read_64_disablecopyonread_adam_m_dense_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp0read_64_disablecopyonread_adam_m_dense_11_kernel^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0p
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  g
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_65/DisableCopyOnReadDisableCopyOnRead0read_65_disablecopyonread_adam_v_dense_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp0read_65_disablecopyonread_adam_v_dense_11_kernel^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0p
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  g
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_66/DisableCopyOnReadDisableCopyOnRead.read_66_disablecopyonread_adam_m_dense_11_bias"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp.read_66_disablecopyonread_adam_m_dense_11_bias^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_67/DisableCopyOnReadDisableCopyOnRead.read_67_disablecopyonread_adam_v_dense_11_bias"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp.read_67_disablecopyonread_adam_v_dense_11_bias^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_68/DisableCopyOnReadDisableCopyOnRead:read_68_disablecopyonread_adam_m_dense_11_p_re_lu_11_alpha"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp:read_68_disablecopyonread_adam_m_dense_11_p_re_lu_11_alpha^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_69/DisableCopyOnReadDisableCopyOnRead:read_69_disablecopyonread_adam_v_dense_11_p_re_lu_11_alpha"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp:read_69_disablecopyonread_adam_v_dense_11_p_re_lu_11_alpha^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_70/DisableCopyOnReadDisableCopyOnRead3read_70_disablecopyonread_adam_m_thicknesses_kernel"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp3read_70_disablecopyonread_adam_m_thicknesses_kernel^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0p
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: g
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_71/DisableCopyOnReadDisableCopyOnRead3read_71_disablecopyonread_adam_v_thicknesses_kernel"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp3read_71_disablecopyonread_adam_v_thicknesses_kernel^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0p
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: g
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_72/DisableCopyOnReadDisableCopyOnRead1read_72_disablecopyonread_adam_m_thicknesses_bias"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp1read_72_disablecopyonread_adam_m_thicknesses_bias^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_73/DisableCopyOnReadDisableCopyOnRead1read_73_disablecopyonread_adam_v_thicknesses_bias"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp1read_73_disablecopyonread_adam_v_thicknesses_bias^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_74/DisableCopyOnReadDisableCopyOnRead!read_74_disablecopyonread_total_6"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp!read_74_disablecopyonread_total_6^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_75/DisableCopyOnReadDisableCopyOnRead!read_75_disablecopyonread_count_6"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp!read_75_disablecopyonread_count_6^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_76/DisableCopyOnReadDisableCopyOnRead!read_76_disablecopyonread_total_5"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp!read_76_disablecopyonread_total_5^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_77/DisableCopyOnReadDisableCopyOnRead!read_77_disablecopyonread_count_5"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp!read_77_disablecopyonread_count_5^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_78/DisableCopyOnReadDisableCopyOnRead!read_78_disablecopyonread_total_4"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOp!read_78_disablecopyonread_total_4^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_79/DisableCopyOnReadDisableCopyOnRead!read_79_disablecopyonread_count_4"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOp!read_79_disablecopyonread_count_4^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_80/DisableCopyOnReadDisableCopyOnRead!read_80_disablecopyonread_total_3"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOp!read_80_disablecopyonread_total_3^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_81/DisableCopyOnReadDisableCopyOnRead!read_81_disablecopyonread_count_3"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOp!read_81_disablecopyonread_count_3^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_82/DisableCopyOnReadDisableCopyOnRead!read_82_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOp!read_82_disablecopyonread_total_2^Read_82/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_83/DisableCopyOnReadDisableCopyOnRead!read_83_disablecopyonread_count_2"/device:CPU:0*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOp!read_83_disablecopyonread_count_2^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_84/DisableCopyOnReadDisableCopyOnRead!read_84_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOp!read_84_disablecopyonread_total_1^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_85/DisableCopyOnReadDisableCopyOnRead!read_85_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOp!read_85_disablecopyonread_count_1^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_86/DisableCopyOnReadDisableCopyOnReadread_86_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOpread_86_disablecopyonread_total^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_87/DisableCopyOnReadDisableCopyOnReadread_87_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOpread_87_disablecopyonread_count^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes
: �$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*�$
value�$B�$YB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*�
value�B�YB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *g
dtypes]
[2Y	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_176Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_177IdentityIdentity_176:output:0^NoOp*
T0*
_output_shapes
: �$
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_177Identity_177:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:Y

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

e
F__inference_dropout_6_layer_call_and_return_conditional_losses_2072276

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
<
input_21
serving_default_input_2:0����������?
first_layer0
StatefulPartitionedCall:0��������� @
second_layer0
StatefulPartitionedCall:1��������� ?
thicknesses0
StatefulPartitionedCall:2���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!
activation

"kernel
#bias"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*
activation

+kernel
,bias"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3_random_generator"
_tf_keras_layer
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:
activation

;kernel
<bias"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
C_random_generator"
_tf_keras_layer
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
J
activation

Kkernel
Lbias"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
S_random_generator"
_tf_keras_layer
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias"
_tf_keras_layer
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
p
activation

qkernel
rbias"
_tf_keras_layer
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
y_random_generator"
_tf_keras_layer
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
�
activation
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
"0
#1
�2
+3
,4
�5
;6
<7
�8
K9
L10
�11
Z12
[13
b14
c15
q16
r17
�18
�19
�20
�21
�22
�23"
trackable_list_wrapper
�
"0
#1
�2
+3
,4
�5
;6
<7
�8
K9
L10
�11
Z12
[13
b14
c15
q16
r17
�18
�19
�20
�21
�22
�23"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
)__inference_model_1_layer_call_fn_2071289
)__inference_model_1_layer_call_fn_2071415
)__inference_model_1_layer_call_fn_2071812
)__inference_model_1_layer_call_fn_2071869�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
D__inference_model_1_layer_call_and_return_conditional_losses_2071068
D__inference_model_1_layer_call_and_return_conditional_losses_2071162
D__inference_model_1_layer_call_and_return_conditional_losses_2072022
D__inference_model_1_layer_call_and_return_conditional_losses_2072140�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
"__inference__wrapped_model_2070648input_2"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
-
�serving_default"
signature_map
6
"0
#1
�2"
trackable_list_wrapper
6
"0
#1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_6_layer_call_fn_2072151�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_6_layer_call_and_return_conditional_losses_2072169�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�alpha"
_tf_keras_layer
": 
��2dense_6/kernel
:�2dense_6/bias
6
+0
,1
�2"
trackable_list_wrapper
6
+0
,1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_7_layer_call_fn_2072180�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_7_layer_call_and_return_conditional_losses_2072198�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�alpha"
_tf_keras_layer
!:	�@2dense_7/kernel
:@2dense_7/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_5_layer_call_fn_2072203
+__inference_dropout_5_layer_call_fn_2072208�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_5_layer_call_and_return_conditional_losses_2072220
F__inference_dropout_5_layer_call_and_return_conditional_losses_2072225�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
6
;0
<1
�2"
trackable_list_wrapper
6
;0
<1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_8_layer_call_fn_2072236�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_8_layer_call_and_return_conditional_losses_2072254�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�alpha"
_tf_keras_layer
 :@ 2dense_8/kernel
: 2dense_8/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_6_layer_call_fn_2072259
+__inference_dropout_6_layer_call_fn_2072264�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_6_layer_call_and_return_conditional_losses_2072276
F__inference_dropout_6_layer_call_and_return_conditional_losses_2072281�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
6
K0
L1
�2"
trackable_list_wrapper
6
K0
L1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_9_layer_call_fn_2072292�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_9_layer_call_and_return_conditional_losses_2072310�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�alpha"
_tf_keras_layer
 :  2dense_9/kernel
: 2dense_9/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_7_layer_call_fn_2072315
+__inference_dropout_7_layer_call_fn_2072320�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_7_layer_call_and_return_conditional_losses_2072332
F__inference_dropout_7_layer_call_and_return_conditional_losses_2072337�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_first_layer_layer_call_fn_2072346�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_first_layer_layer_call_and_return_conditional_losses_2072357�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"  2first_layer/kernel
: 2first_layer/bias
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_second_layer_layer_call_fn_2072366�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_second_layer_layer_call_and_return_conditional_losses_2072377�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#  2second_layer/kernel
: 2second_layer/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_concatenate_1_layer_call_fn_2072384�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2072392�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
6
q0
r1
�2"
trackable_list_wrapper
6
q0
r1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_10_layer_call_fn_2072403�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_10_layer_call_and_return_conditional_losses_2072421�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�alpha"
_tf_keras_layer
!:` 2dense_10/kernel
: 2dense_10/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_8_layer_call_fn_2072426
+__inference_dropout_8_layer_call_fn_2072431�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_8_layer_call_and_return_conditional_losses_2072443
F__inference_dropout_8_layer_call_and_return_conditional_losses_2072448�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
8
�0
�1
�2"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_11_layer_call_fn_2072459�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_11_layer_call_and_return_conditional_losses_2072477�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�alpha"
_tf_keras_layer
!:  2dense_11/kernel
: 2dense_11/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_9_layer_call_fn_2072482
+__inference_dropout_9_layer_call_fn_2072487�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_9_layer_call_and_return_conditional_losses_2072499
F__inference_dropout_9_layer_call_and_return_conditional_losses_2072504�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_thicknesses_layer_call_fn_2072513�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_thicknesses_layer_call_and_return_conditional_losses_2072524�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:" 2thicknesses/kernel
:2thicknesses/bias
&:$�2dense_6/p_re_lu_6/alpha
%:#@2dense_7/p_re_lu_7/alpha
%:# 2dense_8/p_re_lu_8/alpha
%:# 2dense_9/p_re_lu_9/alpha
':% 2dense_10/p_re_lu_10/alpha
':% 2dense_11/p_re_lu_11/alpha
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_model_1_layer_call_fn_2071289input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_1_layer_call_fn_2071415input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_1_layer_call_fn_2071812inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_1_layer_call_fn_2071869inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_1_layer_call_and_return_conditional_losses_2071068input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_1_layer_call_and_return_conditional_losses_2071162input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_1_layer_call_and_return_conditional_losses_2072022inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_1_layer_call_and_return_conditional_losses_2072140inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
%__inference_signature_wrapper_2071755input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
!0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_6_layer_call_fn_2072151inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_6_layer_call_and_return_conditional_losses_2072169inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_p_re_lu_6_layer_call_fn_2072531�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_p_re_lu_6_layer_call_and_return_conditional_losses_2072543�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_7_layer_call_fn_2072180inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_7_layer_call_and_return_conditional_losses_2072198inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_p_re_lu_7_layer_call_fn_2072550�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_p_re_lu_7_layer_call_and_return_conditional_losses_2072562�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
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
�B�
+__inference_dropout_5_layer_call_fn_2072203inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_5_layer_call_fn_2072208inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_5_layer_call_and_return_conditional_losses_2072220inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_5_layer_call_and_return_conditional_losses_2072225inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_8_layer_call_fn_2072236inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_8_layer_call_and_return_conditional_losses_2072254inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_p_re_lu_8_layer_call_fn_2072569�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_p_re_lu_8_layer_call_and_return_conditional_losses_2072581�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
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
�B�
+__inference_dropout_6_layer_call_fn_2072259inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_6_layer_call_fn_2072264inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_6_layer_call_and_return_conditional_losses_2072276inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_6_layer_call_and_return_conditional_losses_2072281inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
J0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_9_layer_call_fn_2072292inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_9_layer_call_and_return_conditional_losses_2072310inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_p_re_lu_9_layer_call_fn_2072588�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_p_re_lu_9_layer_call_and_return_conditional_losses_2072600�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
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
�B�
+__inference_dropout_7_layer_call_fn_2072315inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_7_layer_call_fn_2072320inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_7_layer_call_and_return_conditional_losses_2072332inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_7_layer_call_and_return_conditional_losses_2072337inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_first_layer_layer_call_fn_2072346inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_first_layer_layer_call_and_return_conditional_losses_2072357inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
.__inference_second_layer_layer_call_fn_2072366inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_second_layer_layer_call_and_return_conditional_losses_2072377inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
/__inference_concatenate_1_layer_call_fn_2072384inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2072392inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
p0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_10_layer_call_fn_2072403inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_10_layer_call_and_return_conditional_losses_2072421inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_p_re_lu_10_layer_call_fn_2072607�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_p_re_lu_10_layer_call_and_return_conditional_losses_2072619�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
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
�B�
+__inference_dropout_8_layer_call_fn_2072426inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_8_layer_call_fn_2072431inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_8_layer_call_and_return_conditional_losses_2072443inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_8_layer_call_and_return_conditional_losses_2072448inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_11_layer_call_fn_2072459inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_11_layer_call_and_return_conditional_losses_2072477inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_p_re_lu_11_layer_call_fn_2072626�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_p_re_lu_11_layer_call_and_return_conditional_losses_2072638�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
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
�B�
+__inference_dropout_9_layer_call_fn_2072482inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_9_layer_call_fn_2072487inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_9_layer_call_and_return_conditional_losses_2072499inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_9_layer_call_and_return_conditional_losses_2072504inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_thicknesses_layer_call_fn_2072513inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_thicknesses_layer_call_and_return_conditional_losses_2072524inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
':%
��2Adam/m/dense_6/kernel
':%
��2Adam/v/dense_6/kernel
 :�2Adam/m/dense_6/bias
 :�2Adam/v/dense_6/bias
+:)�2Adam/m/dense_6/p_re_lu_6/alpha
+:)�2Adam/v/dense_6/p_re_lu_6/alpha
&:$	�@2Adam/m/dense_7/kernel
&:$	�@2Adam/v/dense_7/kernel
:@2Adam/m/dense_7/bias
:@2Adam/v/dense_7/bias
*:(@2Adam/m/dense_7/p_re_lu_7/alpha
*:(@2Adam/v/dense_7/p_re_lu_7/alpha
%:#@ 2Adam/m/dense_8/kernel
%:#@ 2Adam/v/dense_8/kernel
: 2Adam/m/dense_8/bias
: 2Adam/v/dense_8/bias
*:( 2Adam/m/dense_8/p_re_lu_8/alpha
*:( 2Adam/v/dense_8/p_re_lu_8/alpha
%:#  2Adam/m/dense_9/kernel
%:#  2Adam/v/dense_9/kernel
: 2Adam/m/dense_9/bias
: 2Adam/v/dense_9/bias
*:( 2Adam/m/dense_9/p_re_lu_9/alpha
*:( 2Adam/v/dense_9/p_re_lu_9/alpha
):'  2Adam/m/first_layer/kernel
):'  2Adam/v/first_layer/kernel
#:! 2Adam/m/first_layer/bias
#:! 2Adam/v/first_layer/bias
*:(  2Adam/m/second_layer/kernel
*:(  2Adam/v/second_layer/kernel
$:" 2Adam/m/second_layer/bias
$:" 2Adam/v/second_layer/bias
&:$` 2Adam/m/dense_10/kernel
&:$` 2Adam/v/dense_10/kernel
 : 2Adam/m/dense_10/bias
 : 2Adam/v/dense_10/bias
,:* 2 Adam/m/dense_10/p_re_lu_10/alpha
,:* 2 Adam/v/dense_10/p_re_lu_10/alpha
&:$  2Adam/m/dense_11/kernel
&:$  2Adam/v/dense_11/kernel
 : 2Adam/m/dense_11/bias
 : 2Adam/v/dense_11/bias
,:* 2 Adam/m/dense_11/p_re_lu_11/alpha
,:* 2 Adam/v/dense_11/p_re_lu_11/alpha
):' 2Adam/m/thicknesses/kernel
):' 2Adam/v/thicknesses/kernel
#:!2Adam/m/thicknesses/bias
#:!2Adam/v/thicknesses/bias
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
�B�
+__inference_p_re_lu_6_layer_call_fn_2072531inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_p_re_lu_6_layer_call_and_return_conditional_losses_2072543inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_p_re_lu_7_layer_call_fn_2072550inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_p_re_lu_7_layer_call_and_return_conditional_losses_2072562inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_p_re_lu_8_layer_call_fn_2072569inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_p_re_lu_8_layer_call_and_return_conditional_losses_2072581inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_p_re_lu_9_layer_call_fn_2072588inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_p_re_lu_9_layer_call_and_return_conditional_losses_2072600inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
,__inference_p_re_lu_10_layer_call_fn_2072607inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_p_re_lu_10_layer_call_and_return_conditional_losses_2072619inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
,__inference_p_re_lu_11_layer_call_fn_2072626inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_p_re_lu_11_layer_call_and_return_conditional_losses_2072638inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
"__inference__wrapped_model_2070648�""#�+,�;<�KL�Z[bcqr������1�.
'�$
"�
input_2����������
� "���
4
first_layer%�"
first_layer��������� 
6
second_layer&�#
second_layer��������� 
4
thicknesses%�"
thicknesses����������
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2072392�~�{
t�q
o�l
"�
inputs_0��������� 
"�
inputs_1��������� 
"�
inputs_2��������� 
� ",�)
"�
tensor_0���������`
� �
/__inference_concatenate_1_layer_call_fn_2072384�~�{
t�q
o�l
"�
inputs_0��������� 
"�
inputs_1��������� 
"�
inputs_2��������� 
� "!�
unknown���������`�
E__inference_dense_10_layer_call_and_return_conditional_losses_2072421eqr�/�,
%�"
 �
inputs���������`
� ",�)
"�
tensor_0��������� 
� �
*__inference_dense_10_layer_call_fn_2072403Zqr�/�,
%�"
 �
inputs���������`
� "!�
unknown��������� �
E__inference_dense_11_layer_call_and_return_conditional_losses_2072477g���/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
*__inference_dense_11_layer_call_fn_2072459\���/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
D__inference_dense_6_layer_call_and_return_conditional_losses_2072169g"#�0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_6_layer_call_fn_2072151\"#�0�-
&�#
!�
inputs����������
� ""�
unknown�����������
D__inference_dense_7_layer_call_and_return_conditional_losses_2072198f+,�0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
)__inference_dense_7_layer_call_fn_2072180[+,�0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
D__inference_dense_8_layer_call_and_return_conditional_losses_2072254e;<�/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0��������� 
� �
)__inference_dense_8_layer_call_fn_2072236Z;<�/�,
%�"
 �
inputs���������@
� "!�
unknown��������� �
D__inference_dense_9_layer_call_and_return_conditional_losses_2072310eKL�/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
)__inference_dense_9_layer_call_fn_2072292ZKL�/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
F__inference_dropout_5_layer_call_and_return_conditional_losses_2072220c3�0
)�&
 �
inputs���������@
p
� ",�)
"�
tensor_0���������@
� �
F__inference_dropout_5_layer_call_and_return_conditional_losses_2072225c3�0
)�&
 �
inputs���������@
p 
� ",�)
"�
tensor_0���������@
� �
+__inference_dropout_5_layer_call_fn_2072203X3�0
)�&
 �
inputs���������@
p
� "!�
unknown���������@�
+__inference_dropout_5_layer_call_fn_2072208X3�0
)�&
 �
inputs���������@
p 
� "!�
unknown���������@�
F__inference_dropout_6_layer_call_and_return_conditional_losses_2072276c3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
F__inference_dropout_6_layer_call_and_return_conditional_losses_2072281c3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
+__inference_dropout_6_layer_call_fn_2072259X3�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
+__inference_dropout_6_layer_call_fn_2072264X3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� �
F__inference_dropout_7_layer_call_and_return_conditional_losses_2072332c3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
F__inference_dropout_7_layer_call_and_return_conditional_losses_2072337c3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
+__inference_dropout_7_layer_call_fn_2072315X3�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
+__inference_dropout_7_layer_call_fn_2072320X3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� �
F__inference_dropout_8_layer_call_and_return_conditional_losses_2072443c3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
F__inference_dropout_8_layer_call_and_return_conditional_losses_2072448c3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
+__inference_dropout_8_layer_call_fn_2072426X3�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
+__inference_dropout_8_layer_call_fn_2072431X3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� �
F__inference_dropout_9_layer_call_and_return_conditional_losses_2072499c3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
F__inference_dropout_9_layer_call_and_return_conditional_losses_2072504c3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
+__inference_dropout_9_layer_call_fn_2072482X3�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
+__inference_dropout_9_layer_call_fn_2072487X3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� �
H__inference_first_layer_layer_call_and_return_conditional_losses_2072357cZ[/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
-__inference_first_layer_layer_call_fn_2072346XZ[/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
D__inference_model_1_layer_call_and_return_conditional_losses_2071068�""#�+,�;<�KL�Z[bcqr������9�6
/�,
"�
input_2����������
p

 
� "�|
u�r
$�!

tensor_0_0��������� 
$�!

tensor_0_1��������� 
$�!

tensor_0_2���������
� �
D__inference_model_1_layer_call_and_return_conditional_losses_2071162�""#�+,�;<�KL�Z[bcqr������9�6
/�,
"�
input_2����������
p 

 
� "�|
u�r
$�!

tensor_0_0��������� 
$�!

tensor_0_1��������� 
$�!

tensor_0_2���������
� �
D__inference_model_1_layer_call_and_return_conditional_losses_2072022�""#�+,�;<�KL�Z[bcqr������8�5
.�+
!�
inputs����������
p

 
� "�|
u�r
$�!

tensor_0_0��������� 
$�!

tensor_0_1��������� 
$�!

tensor_0_2���������
� �
D__inference_model_1_layer_call_and_return_conditional_losses_2072140�""#�+,�;<�KL�Z[bcqr������8�5
.�+
!�
inputs����������
p 

 
� "�|
u�r
$�!

tensor_0_0��������� 
$�!

tensor_0_1��������� 
$�!

tensor_0_2���������
� �
)__inference_model_1_layer_call_fn_2071289�""#�+,�;<�KL�Z[bcqr������9�6
/�,
"�
input_2����������
p

 
� "o�l
"�
tensor_0��������� 
"�
tensor_1��������� 
"�
tensor_2����������
)__inference_model_1_layer_call_fn_2071415�""#�+,�;<�KL�Z[bcqr������9�6
/�,
"�
input_2����������
p 

 
� "o�l
"�
tensor_0��������� 
"�
tensor_1��������� 
"�
tensor_2����������
)__inference_model_1_layer_call_fn_2071812�""#�+,�;<�KL�Z[bcqr������8�5
.�+
!�
inputs����������
p

 
� "o�l
"�
tensor_0��������� 
"�
tensor_1��������� 
"�
tensor_2����������
)__inference_model_1_layer_call_fn_2071869�""#�+,�;<�KL�Z[bcqr������8�5
.�+
!�
inputs����������
p 

 
� "o�l
"�
tensor_0��������� 
"�
tensor_1��������� 
"�
tensor_2����������
G__inference_p_re_lu_10_layer_call_and_return_conditional_losses_2072619l�8�5
.�+
)�&
inputs������������������
� ",�)
"�
tensor_0��������� 
� �
,__inference_p_re_lu_10_layer_call_fn_2072607a�8�5
.�+
)�&
inputs������������������
� "!�
unknown��������� �
G__inference_p_re_lu_11_layer_call_and_return_conditional_losses_2072638l�8�5
.�+
)�&
inputs������������������
� ",�)
"�
tensor_0��������� 
� �
,__inference_p_re_lu_11_layer_call_fn_2072626a�8�5
.�+
)�&
inputs������������������
� "!�
unknown��������� �
F__inference_p_re_lu_6_layer_call_and_return_conditional_losses_2072543m�8�5
.�+
)�&
inputs������������������
� "-�*
#� 
tensor_0����������
� �
+__inference_p_re_lu_6_layer_call_fn_2072531b�8�5
.�+
)�&
inputs������������������
� ""�
unknown�����������
F__inference_p_re_lu_7_layer_call_and_return_conditional_losses_2072562l�8�5
.�+
)�&
inputs������������������
� ",�)
"�
tensor_0���������@
� �
+__inference_p_re_lu_7_layer_call_fn_2072550a�8�5
.�+
)�&
inputs������������������
� "!�
unknown���������@�
F__inference_p_re_lu_8_layer_call_and_return_conditional_losses_2072581l�8�5
.�+
)�&
inputs������������������
� ",�)
"�
tensor_0��������� 
� �
+__inference_p_re_lu_8_layer_call_fn_2072569a�8�5
.�+
)�&
inputs������������������
� "!�
unknown��������� �
F__inference_p_re_lu_9_layer_call_and_return_conditional_losses_2072600l�8�5
.�+
)�&
inputs������������������
� ",�)
"�
tensor_0��������� 
� �
+__inference_p_re_lu_9_layer_call_fn_2072588a�8�5
.�+
)�&
inputs������������������
� "!�
unknown��������� �
I__inference_second_layer_layer_call_and_return_conditional_losses_2072377cbc/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
.__inference_second_layer_layer_call_fn_2072366Xbc/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
%__inference_signature_wrapper_2071755�""#�+,�;<�KL�Z[bcqr������<�9
� 
2�/
-
input_2"�
input_2����������"���
4
first_layer%�"
first_layer��������� 
6
second_layer&�#
second_layer��������� 
4
thicknesses%�"
thicknesses����������
H__inference_thicknesses_layer_call_and_return_conditional_losses_2072524e��/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
-__inference_thicknesses_layer_call_fn_2072513Z��/�,
%�"
 �
inputs��������� 
� "!�
unknown���������