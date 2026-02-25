from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataMap(_message.Message):
    __slots__ = ["data"]
    class DataEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: PairList
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[PairList, _Mapping]] = ...) -> None: ...
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.MessageMap[str, PairList]
    def __init__(self, data: _Optional[_Mapping[str, PairList]] = ...) -> None: ...

class Pair(_message.Message):
    __slots__ = ["srcSentence", "tgtSentence"]
    SRCSENTENCE_FIELD_NUMBER: _ClassVar[int]
    TGTSENTENCE_FIELD_NUMBER: _ClassVar[int]
    srcSentence: int
    tgtSentence: int
    def __init__(self, srcSentence: _Optional[int] = ..., tgtSentence: _Optional[int] = ...) -> None: ...

class PairList(_message.Message):
    __slots__ = ["items"]
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[Pair]
    def __init__(self, items: _Optional[_Iterable[_Union[Pair, _Mapping]]] = ...) -> None: ...
