# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: orbb_tracking.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import orbb_definitions_pb2 as orbb__definitions__pb2
import orbb_metadata_pb2 as orbb__metadata__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='orbb_tracking.proto',
  package='orbb',
  syntax='proto2',
  serialized_pb=_b('\n\x13orbb_tracking.proto\x12\x04orbb\x1a\x16orbb_definitions.proto\x1a\x13orbb_metadata.proto\"\xa0\x01\n\nRealObject\x12\x18\n\x04type\x18\x01 \x02(\x0e\x32\n.orbb.Mark\x12\x1c\n\x06region\x18\x02 \x02(\x0e\x32\x0c.orbb.Region\x12\r\n\x05width\x18\x03 \x02(\x02\x12\x0e\n\x06height\x18\x04 \x02(\x02\x12\x1b\n\x13trunk_size_increase\x18\x05 \x01(\x02\x12\x1e\n\x16motorway_size_increase\x18\x06 \x01(\x02\">\n\x12TrackingConfigMeta\x12(\n\x0etrackable_objs\x18\x01 \x03(\x0b\x32\x10.orbb.RealObject')
  ,
  dependencies=[orbb__definitions__pb2.DESCRIPTOR,orbb__metadata__pb2.DESCRIPTOR,])




_REALOBJECT = _descriptor.Descriptor(
  name='RealObject',
  full_name='orbb.RealObject',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='orbb.RealObject.type', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='region', full_name='orbb.RealObject.region', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='width', full_name='orbb.RealObject.width', index=2,
      number=3, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='height', full_name='orbb.RealObject.height', index=3,
      number=4, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='trunk_size_increase', full_name='orbb.RealObject.trunk_size_increase', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='motorway_size_increase', full_name='orbb.RealObject.motorway_size_increase', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=75,
  serialized_end=235,
)


_TRACKINGCONFIGMETA = _descriptor.Descriptor(
  name='TrackingConfigMeta',
  full_name='orbb.TrackingConfigMeta',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='trackable_objs', full_name='orbb.TrackingConfigMeta.trackable_objs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=237,
  serialized_end=299,
)

_REALOBJECT.fields_by_name['type'].enum_type = orbb__definitions__pb2._MARK
_REALOBJECT.fields_by_name['region'].enum_type = orbb__definitions__pb2._REGION
_TRACKINGCONFIGMETA.fields_by_name['trackable_objs'].message_type = _REALOBJECT
DESCRIPTOR.message_types_by_name['RealObject'] = _REALOBJECT
DESCRIPTOR.message_types_by_name['TrackingConfigMeta'] = _TRACKINGCONFIGMETA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RealObject = _reflection.GeneratedProtocolMessageType('RealObject', (_message.Message,), dict(
  DESCRIPTOR = _REALOBJECT,
  __module__ = 'orbb_tracking_pb2'
  # @@protoc_insertion_point(class_scope:orbb.RealObject)
  ))
_sym_db.RegisterMessage(RealObject)

TrackingConfigMeta = _reflection.GeneratedProtocolMessageType('TrackingConfigMeta', (_message.Message,), dict(
  DESCRIPTOR = _TRACKINGCONFIGMETA,
  __module__ = 'orbb_tracking_pb2'
  # @@protoc_insertion_point(class_scope:orbb.TrackingConfigMeta)
  ))
_sym_db.RegisterMessage(TrackingConfigMeta)


# @@protoc_insertion_point(module_scope)
