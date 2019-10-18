extern crate glsl;

extern crate rspirv;
extern crate spirv_headers as spirv;

use rspirv::binary::Assemble;
use rspirv::binary::Disassemble;

use glsl::parser::Parse;
use glsl::syntax::TranslationUnit;
use rspirv::mr::{Builder, Operand};
use spirv::Word;
use std::collections::HashMap;

mod hir;

use hir::State;
use hir::Type;

use std::io::Write as IoWrite;

pub fn glsl_to_spirv(input: &str) -> String {
  let ast = TranslationUnit::parse(input).unwrap();
  let mut state = hir::State::new();
  let hir = hir::ast_to_hir(&mut state, &ast);
  let mut b = rspirv::mr::Builder::new();
  b.capability(spirv::Capability::Shader);
  b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);


  let mut state = OutputState {
    hir: state,
    model: spirv::ExecutionModel::Fragment,
    return_type: None,
    return_declared: false,
    builder: b,
    emitted_types: HashMap::new(),
    emitted_syms: HashMap::new(),
  };

  translate_translation_unit(&mut state, &hir);

  let b = state.builder;

  let module = b.module();

  // Assembling
  let code = module.assemble();
  assert!(code.len() > 20);  // Module header contains 5 words
  assert_eq!(spirv::MAGIC_NUMBER, code[0]);

  // Parsing
  let mut loader = rspirv::mr::Loader::new();
  rspirv::binary::parse_words(&code, &mut loader).unwrap();
  let module = loader.module();

  // Disassembling
  module.disassemble()
}

fn main() {
  let r = TranslationUnit::parse("layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0, 0.0, 0.0, 1.0);
}");

    /*void main() {
      vec3 p = vec3(0, 1, 4);
      float x = 1. * .5 + p.r;
    }");

*/
  let r = r.unwrap();
  println!("{:?}", r);

  let mut state = hir::State::new();
  //println!("{:#?}", r);
  let hir = hir::ast_to_hir(&mut state, &r);
  //println!("{:#?}", hir);
  // Building
  let mut b = rspirv::mr::Builder::new();
  b.capability(spirv::Capability::Shader);
  b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);


  let mut state = OutputState {
    hir: state,
    model: spirv::ExecutionModel::Fragment,
    return_type: None,
    return_declared: false,
    builder: b,
    emitted_types: HashMap::new(),
    emitted_syms: HashMap::new(),
  };

  translate_translation_unit(&mut state, &hir);

  let b = state.builder;

  let module = b.module();

  // Assembling
  let code = module.assemble();
  assert!(code.len() > 20);  // Module header contains 5 words
  assert_eq!(spirv::MAGIC_NUMBER, code[0]);
  let mut f = std::fs::File::create("out.frag").unwrap();
  let len = code[..].len() * 4;
  let buf = unsafe { std::slice::from_raw_parts(code[..].as_ptr() as *const u8, len) };
  f.write(buf);

  // Parsing
  let mut loader = rspirv::mr::Loader::new();
  rspirv::binary::parse_words(&code, &mut loader).unwrap();
  let module = loader.module();

  // Disassembling
  println!("{}", module.disassemble());
}

pub struct Variable {
  location: Word,
  ty: Word,
}

pub struct OutputState {
  builder: rspirv::mr::Builder,
  model: spirv::ExecutionModel,
  hir: hir::State,
  return_type: Option<Box<hir::Type>>,
  return_declared: bool,
  //XXX: we can probably hash on something better than String
  emitted_types: HashMap<String, Word>,
  emitted_syms: HashMap<hir::SymRef, Variable>,
}

use std::fmt::Write;

use glsl::syntax;

pub fn show_identifier<F>(f: &mut F, i: &syntax::Identifier) where F: Write {
  let _ = f.write_str(&i.0);
}

pub fn show_sym<F>(f: &mut F, state: &mut OutputState, i: &hir::SymRef) where F: Write {
  let name = state.hir.sym(*i).name.as_str();
  let _ = f.write_str(name);
}

pub fn show_type_name<F>(f: &mut F, t: &syntax::TypeName) where F: Write {
  let _ = f.write_str(&t.0);
}

pub fn show_type_kind<F>(f: &mut F, state: &mut OutputState, t: &hir::TypeKind) where F: Write {
  match *t {
    hir::TypeKind::Void => { let _ = f.write_str("void"); }
    hir::TypeKind::Bool => { let _ = f.write_str("bool"); }
    hir::TypeKind::Int => { let _ = f.write_str("int"); }
    hir::TypeKind::UInt => { let _ = f.write_str("uint"); }
    hir::TypeKind::Float => { let _ = f.write_str("float"); }
    hir::TypeKind::Double => { let _ = f.write_str("double"); }
    hir::TypeKind::Vec2 => { let _ = f.write_str("vec2"); }
    hir::TypeKind::Vec3 => { let _ = f.write_str("vec3"); }
    hir::TypeKind::Vec4 => { let _ = f.write_str("vec4"); }
    hir::TypeKind::DVec2 => { let _ = f.write_str("dvec2"); }
    hir::TypeKind::DVec3 => { let _ = f.write_str("dvec3"); }
    hir::TypeKind::DVec4 => { let _ = f.write_str("dvec4"); }
    hir::TypeKind::BVec2 => { let _ = f.write_str("bvec2"); }
    hir::TypeKind::BVec3 => { let _ = f.write_str("bvec3"); }
    hir::TypeKind::BVec4 => { let _ = f.write_str("bvec4"); }
    hir::TypeKind::IVec2 => { let _ = f.write_str("ivec2"); }
    hir::TypeKind::IVec3 => { let _ = f.write_str("ivec3"); }
    hir::TypeKind::IVec4 => { let _ = f.write_str("ivec4"); }
    hir::TypeKind::UVec2 => { let _ = f.write_str("uvec2"); }
    hir::TypeKind::UVec3 => { let _ = f.write_str("uvec3"); }
    hir::TypeKind::UVec4 => { let _ = f.write_str("uvec4"); }
    hir::TypeKind::Mat2 => { let _ = f.write_str("mat2"); }
    hir::TypeKind::Mat3 => { let _ = f.write_str("mat3"); }
    hir::TypeKind::Mat4 => { let _ = f.write_str("mat4"); }
    hir::TypeKind::Mat23 => { let _ = f.write_str("mat23"); }
    hir::TypeKind::Mat24 => { let _ = f.write_str("mat24"); }
    hir::TypeKind::Mat32 => { let _ = f.write_str("mat32"); }
    hir::TypeKind::Mat34 => { let _ = f.write_str("mat34"); }
    hir::TypeKind::Mat42 => { let _ = f.write_str("mat42"); }
    hir::TypeKind::Mat43 => { let _ = f.write_str("mat43"); }
    hir::TypeKind::DMat2 => { let _ = f.write_str("dmat2"); }
    hir::TypeKind::DMat3 => { let _ = f.write_str("dmat3"); }
    hir::TypeKind::DMat4 => { let _ = f.write_str("dmat4"); }
    hir::TypeKind::DMat23 => { let _ = f.write_str("dmat23"); }
    hir::TypeKind::DMat24 => { let _ = f.write_str("dmat24"); }
    hir::TypeKind::DMat32 => { let _ = f.write_str("dmat32"); }
    hir::TypeKind::DMat34 => { let _ = f.write_str("dmat34"); }
    hir::TypeKind::DMat42 => { let _ = f.write_str("dmat42"); }
    hir::TypeKind::DMat43 => { let _ = f.write_str("dmat43"); }
    hir::TypeKind::Sampler1D => { let _ = f.write_str("sampler1D"); }
    hir::TypeKind::Image1D => { let _ = f.write_str("image1D"); }
    hir::TypeKind::Sampler2D => { let _ = f.write_str("sampler2D"); }
    hir::TypeKind::Image2D => { let _ = f.write_str("image2D"); }
    hir::TypeKind::Sampler3D => { let _ = f.write_str("sampler3D"); }
    hir::TypeKind::Image3D => { let _ = f.write_str("image3D"); }
    hir::TypeKind::SamplerCube => { let _ = f.write_str("samplerCube"); }
    hir::TypeKind::ImageCube => { let _ = f.write_str("imageCube"); }
    hir::TypeKind::Sampler2DRect => { let _ = f.write_str("sampler2DRect"); }
    hir::TypeKind::Image2DRect => { let _ = f.write_str("image2DRect"); }
    hir::TypeKind::Sampler1DArray => { let _ = f.write_str("sampler1DArray"); }
    hir::TypeKind::Image1DArray => { let _ = f.write_str("image1DArray"); }
    hir::TypeKind::Sampler2DArray => { let _ = f.write_str("sampler2DArray"); }
    hir::TypeKind::Image2DArray => { let _ = f.write_str("image2DArray"); }
    hir::TypeKind::SamplerBuffer => { let _ = f.write_str("samplerBuffer"); }
    hir::TypeKind::ImageBuffer => { let _ = f.write_str("imageBuffer"); }
    hir::TypeKind::Sampler2DMS => { let _ = f.write_str("sampler2DMS"); }
    hir::TypeKind::Image2DMS => { let _ = f.write_str("image2DMS"); }
    hir::TypeKind::Sampler2DMSArray => { let _ = f.write_str("sampler2DMSArray"); }
    hir::TypeKind::Image2DMSArray => { let _ = f.write_str("image2DMSArray"); }
    hir::TypeKind::SamplerCubeArray => { let _ = f.write_str("samplerCubeArray"); }
    hir::TypeKind::ImageCubeArray => { let _ = f.write_str("imageCubeArray"); }
    hir::TypeKind::Sampler1DShadow => { let _ = f.write_str("sampler1DShadow"); }
    hir::TypeKind::Sampler2DShadow => { let _ = f.write_str("sampler2DShadow"); }
    hir::TypeKind::Sampler2DRectShadow => { let _ = f.write_str("sampler2DRectShadow"); }
    hir::TypeKind::Sampler1DArrayShadow => { let _ = f.write_str("sampler1DArrayShadow"); }
    hir::TypeKind::Sampler2DArrayShadow => { let _ = f.write_str("sampler2DArrayShadow"); }
    hir::TypeKind::SamplerCubeShadow => { let _ = f.write_str("samplerCubeShadow"); }
    hir::TypeKind::SamplerCubeArrayShadow => { let _ = f.write_str("samplerCubeArrayShadow"); }
    hir::TypeKind::ISampler1D => { let _ = f.write_str("isampler1D"); }
    hir::TypeKind::IImage1D => { let _ = f.write_str("iimage1D"); }
    hir::TypeKind::ISampler2D => { let _ = f.write_str("isampler2D"); }
    hir::TypeKind::IImage2D => { let _ = f.write_str("iimage2D"); }
    hir::TypeKind::ISampler3D => { let _ = f.write_str("isampler3D"); }
    hir::TypeKind::IImage3D => { let _ = f.write_str("iimage3D"); }
    hir::TypeKind::ISamplerCube => { let _ = f.write_str("isamplerCube"); }
    hir::TypeKind::IImageCube => { let _ = f.write_str("iimageCube"); }
    hir::TypeKind::ISampler2DRect => { let _ = f.write_str("isampler2DRect"); }
    hir::TypeKind::IImage2DRect => { let _ = f.write_str("iimage2DRect"); }
    hir::TypeKind::ISampler1DArray => { let _ = f.write_str("isampler1DArray"); }
    hir::TypeKind::IImage1DArray => { let _ = f.write_str("iimage1DArray"); }
    hir::TypeKind::ISampler2DArray => { let _ = f.write_str("isampler2DArray"); }
    hir::TypeKind::IImage2DArray => { let _ = f.write_str("iimage2DArray"); }
    hir::TypeKind::ISamplerBuffer => { let _ = f.write_str("isamplerBuffer"); }
    hir::TypeKind::IImageBuffer => { let _ = f.write_str("iimageBuffer"); }
    hir::TypeKind::ISampler2DMS => { let _ = f.write_str("isampler2MS"); }
    hir::TypeKind::IImage2DMS => { let _ = f.write_str("iimage2DMS"); }
    hir::TypeKind::ISampler2DMSArray => { let _ = f.write_str("isampler2DMSArray"); }
    hir::TypeKind::IImage2DMSArray => { let _ = f.write_str("iimage2DMSArray"); }
    hir::TypeKind::ISamplerCubeArray => { let _ = f.write_str("isamplerCubeArray"); }
    hir::TypeKind::IImageCubeArray => { let _ = f.write_str("iimageCubeArray"); }
    hir::TypeKind::AtomicUInt => { let _ = f.write_str("atomic_uint"); }
    hir::TypeKind::USampler1D => { let _ = f.write_str("usampler1D"); }
    hir::TypeKind::UImage1D => { let _ = f.write_str("uimage1D"); }
    hir::TypeKind::USampler2D => { let _ = f.write_str("usampler2D"); }
    hir::TypeKind::UImage2D => { let _ = f.write_str("uimage2D"); }
    hir::TypeKind::USampler3D => { let _ = f.write_str("usampler3D"); }
    hir::TypeKind::UImage3D => { let _ = f.write_str("uimage3D"); }
    hir::TypeKind::USamplerCube => { let _ = f.write_str("usamplerCube"); }
    hir::TypeKind::UImageCube => { let _ = f.write_str("uimageCube"); }
    hir::TypeKind::USampler2DRect => { let _ = f.write_str("usampler2DRect"); }
    hir::TypeKind::UImage2DRect => { let _ = f.write_str("uimage2DRect"); }
    hir::TypeKind::USampler1DArray => { let _ = f.write_str("usampler1DArray"); }
    hir::TypeKind::UImage1DArray => { let _ = f.write_str("uimage1DArray"); }
    hir::TypeKind::USampler2DArray => { let _ = f.write_str("usampler2DArray"); }
    hir::TypeKind::UImage2DArray => { let _ = f.write_str("uimage2DArray"); }
    hir::TypeKind::USamplerBuffer => { let _ = f.write_str("usamplerBuffer"); }
    hir::TypeKind::UImageBuffer => { let _ = f.write_str("uimageBuffer"); }
    hir::TypeKind::USampler2DMS => { let _ = f.write_str("usampler2DMS"); }
    hir::TypeKind::UImage2DMS => { let _ = f.write_str("uimage2DMS"); }
    hir::TypeKind::USampler2DMSArray => { let _ = f.write_str("usamplerDMSArray"); }
    hir::TypeKind::UImage2DMSArray => { let _ = f.write_str("uimage2DMSArray"); }
    hir::TypeKind::USamplerCubeArray => { let _ = f.write_str("usamplerCubeArray"); }
    hir::TypeKind::UImageCubeArray => { let _ = f.write_str("uimageCubeArray"); }
    hir::TypeKind::Struct(ref s) => panic!(),
  }
}

pub fn show_type_specifier<F>(f: &mut F, state: &mut OutputState, t: &syntax::TypeSpecifier) where F: Write {
  /*show_type_specifier_non_array(f, state, &t.ty);

  if let Some(ref arr_spec) = t.array_specifier {
    show_array_spec(f, arr_spec);
  }*/
}

pub fn show_type<F>(f: &mut F, state: &mut OutputState, t: &Type) where F: Write {

  if let Some(ref precision) = t.precision {
    show_precision_qualifier(f, precision);
    let _ = f.write_str(" ");
  }

  show_type_kind(f, state, &t.kind);

  if let Some(ref arr_spec) = t.array_sizes {
    panic!();
  }
}

pub fn show_struct_non_declaration<F>(f: &mut F, state: &mut OutputState, s: &syntax::StructSpecifier) where F: Write {
  let _ = f.write_str("struct ");

  if let Some(ref name) = s.name {
    let _ = write!(f, "{} ", name);
  }

  let _ = f.write_str("{\n");

  for field in &s.fields.0 {
    show_struct_field(f, state, field);
  }

  let _ = f.write_str("}");
}

pub fn show_struct<F>(f: &mut F, state: &mut OutputState, s: &syntax::StructSpecifier) where F: Write {
  show_struct_non_declaration(f, state, s);
  let _ = f.write_str(";\n");
}


pub fn show_struct_field<F>(f: &mut F, state: &mut OutputState, field: &syntax::StructFieldSpecifier) where F: Write {
  /*if let Some(ref qual) = field.qualifier {
    show_type_qualifier(f, &qual);
    let _ = f.write_str(" ");
  }

  show_type_specifier(f, state, &field.ty);
  let _ = f.write_str(" ");

  // there’s at least one identifier
  let mut identifiers = field.identifiers.0.iter();
  let identifier = identifiers.next().unwrap();

  show_arrayed_identifier(f, identifier);

  // write the rest of the identifiers
  for identifier in identifiers {
    let _ = f.write_str(", ");
    show_arrayed_identifier(f, identifier);
  }

  let _ = f.write_str(";\n");*/
}

pub fn show_array_sizes<F>(f: &mut F, a: &hir::ArraySizes) where F: Write {
  panic!()
  /*
  match *a {
    syntax::ArraySpecifier::Unsized => { let _ = f.write_str("[]"); }
    syntax::ArraySpecifier::ExplicitlySized(ref e) => {
      let _ = f.write_str("[");
      show_expr(f, &e);
      let _ = f.write_str("]");
    }
  }*/
}

pub fn show_arrayed_identifier<F>(f: &mut F, ident: &syntax::Identifier, ty: &hir::Type) where F: Write {
  let _ = write!(f, "{}", ident);

  if let Some(ref arr_spec) = ty.array_sizes {
    show_array_sizes(f, &arr_spec);
  }
}

pub fn show_type_qualifier<F>(f: &mut F, q: &hir::TypeQualifier) where F: Write {
  let mut qualifiers = q.qualifiers.0.iter();
  let first = qualifiers.next().unwrap();

  show_type_qualifier_spec(f, first);

  for qual_spec in qualifiers {
    let _ = f.write_str(" ");
    show_type_qualifier_spec(f, qual_spec)
  }
}

pub fn show_type_qualifier_spec<F>(f: &mut F, q: &hir::TypeQualifierSpec) where F: Write {
  match *q {
    hir::TypeQualifierSpec::Layout(ref l) => show_layout_qualifier(f, &l),
    hir::TypeQualifierSpec::Interpolation(ref i) => show_interpolation_qualifier(f, &i),
    hir::TypeQualifierSpec::Invariant => { let _ = f.write_str("invariant"); },
    hir::TypeQualifierSpec::Precise => { let _ = f.write_str("precise"); }
    hir::TypeQualifierSpec::Memory(_) => { panic!() }
    hir::TypeQualifierSpec::Parameter(_) => { panic!() }
  }
}

pub fn show_storage_qualifier<F>(f: &mut F, q: &syntax::StorageQualifier) where F: Write {
  match *q {
    syntax::StorageQualifier::Const => { let _ = f.write_str("const"); }
    syntax::StorageQualifier::InOut => { let _ = f.write_str("inout"); }
    syntax::StorageQualifier::In => { let _ = f.write_str("in"); }
    syntax::StorageQualifier::Out => { let _ = f.write_str("out"); }
    syntax::StorageQualifier::Centroid => { let _ = f.write_str("centroid"); }
    syntax::StorageQualifier::Patch => { let _ = f.write_str("patch"); }
    syntax::StorageQualifier::Sample => { let _ = f.write_str("sample"); }
    syntax::StorageQualifier::Uniform => { let _ = f.write_str("uniform"); }
    syntax::StorageQualifier::Buffer => { let _ = f.write_str("buffer"); }
    syntax::StorageQualifier::Shared => { let _ = f.write_str("shared"); }
    syntax::StorageQualifier::Coherent => { let _ = f.write_str("coherent"); }
    syntax::StorageQualifier::Volatile => { let _ = f.write_str("volatile"); }
    syntax::StorageQualifier::Restrict => { let _ = f.write_str("restrict"); }
    syntax::StorageQualifier::ReadOnly => { let _ = f.write_str("readonly"); }
    syntax::StorageQualifier::WriteOnly => { let _ = f.write_str("writeonly"); }
    syntax::StorageQualifier::Subroutine(ref n) => show_subroutine(f, &n)
  }
}

pub fn show_subroutine<F>(f: &mut F, types: &Vec<syntax::TypeName>) where F: Write {
  let _ = f.write_str("subroutine");

  if !types.is_empty() {
    let _ = f.write_str("(");

    let mut types_iter = types.iter();
    let first = types_iter.next().unwrap();

    show_type_name(f, first);

    for type_name in types_iter {
      let _ = f.write_str(", ");
      show_type_name(f, type_name);
    }

    let _ = f.write_str(")");
  }
}

pub fn show_layout_qualifier<F>(f: &mut F, l: &syntax::LayoutQualifier) where F: Write {
  let mut qualifiers = l.ids.0.iter();
  let first = qualifiers.next().unwrap();

  let _ = f.write_str("layout (");
  show_layout_qualifier_spec(f, first);

  for qual_spec in qualifiers {
    let _ = f.write_str(", ");
    show_layout_qualifier_spec(f, qual_spec);
  }

  let _ = f.write_str(")");
}

pub fn show_layout_qualifier_spec<F>(f: &mut F, l: &syntax::LayoutQualifierSpec) where F: Write {
  match *l {
    syntax::LayoutQualifierSpec::Identifier(ref i, Some(ref e)) => {
      let _ = write!(f, "{} = ", i);
      show_expr(f, &e);
    }
    syntax::LayoutQualifierSpec::Identifier(ref i, None) => show_identifier(f, &i),
    syntax::LayoutQualifierSpec::Shared => { let _ = f.write_str("shared"); }
  }
}

pub fn show_precision_qualifier<F>(f: &mut F, p: &syntax::PrecisionQualifier) where F: Write {
  match *p {
    syntax::PrecisionQualifier::High => { let _ = f.write_str("highp"); }
    syntax::PrecisionQualifier::Medium => { let _ = f.write_str("mediump"); }
    syntax::PrecisionQualifier::Low => { let _ = f.write_str("low"); }
  }
}

pub fn show_interpolation_qualifier<F>(f: &mut F, i: &syntax::InterpolationQualifier) where F: Write {
  match *i {
    syntax::InterpolationQualifier::Smooth => { let _ = f.write_str("smooth"); }
    syntax::InterpolationQualifier::Flat => { let _ = f.write_str("flat"); }
    syntax::InterpolationQualifier::NoPerspective => { let _ = f.write_str("noperspective"); }
  }
}

pub fn show_float<F>(f: &mut F, x: f32) where F: Write {
  if x.fract() == 0. {
    let _ = write!(f, "{}.", x);
  } else {
    let _ = write!(f, "{}", x);
  }
}

pub fn show_double<F>(f: &mut F, x: f64) where F: Write {
  if x.fract() == 0. {
    let _ = write!(f, "{}.", x);
  } else {
    let _ = write!(f, "{}", x);
  }
}

pub fn emit_float(state: &mut OutputState) -> Word {
  match state.emitted_types.get("float") {
    Some(t) => *t,
    None => {
      let float = state.builder.type_float(32);
      state.emitted_types.insert("float".to_string(), float);
      float
    }
  }
}

pub fn emit_type(state: &mut OutputState, ty: &hir::Type) -> Word {
  if ty.precision.is_some() {
    panic!()
  }
  if ty.array_sizes.is_some() {
    panic!()
  }
  match ty.kind {
    hir::TypeKind::Float => {
      emit_float(state)
    }
    hir::TypeKind::Double => {
      //XXX: actually use double here
      emit_float(state)
    }
    hir::TypeKind::Vec4 => {
      emit_vec4(state)
    }
    _ => panic!("{:?}", ty.kind)
  }
}

pub fn emit_vec4(state: &mut OutputState) -> Word {
  match state.emitted_types.get("vec4") {
    Some(t) => *t,
    None => {
      let float = emit_float(state);
      let float_vec4 = state.builder.type_vector(float, 4);
      state.emitted_types.insert("vec4".to_string(), float_vec4);
      float_vec4
    }
  }
}

pub fn emit_void(state: &mut OutputState) -> Word {
  match state.emitted_types.get("void") {
    Some(t) => *t,
    None => {
      let void = state.builder.type_void();
      state.emitted_types.insert("void".to_string(), void);
      void
    }
  }
}

pub fn emit_sym(state: &mut OutputState, s: hir::SymRef) -> Word {
  match state.emitted_syms.get(&s) {
    Some(s) => s.location,
    None => {
      let name = &state.hir.sym(s).name;
      match name.as_ref() {
        "gl_FragColor" => {
          // XXX: we emit these special variables lazily
          // we should do better than matching by name
          let float_vec4 = emit_vec4(state);
          let b = &mut state.builder;
          let output = b.type_pointer(None, spirv::StorageClass::Output, float_vec4);
          let output_var = b.variable(output, None, spirv::StorageClass::Output, None);
          b.decorate(output_var, spirv::Decoration::Location, [Operand::LiteralInt32(0)]);
          state.emitted_syms.insert(s, Variable { location: output_var, ty: output});
          output_var
        }
        _ => panic!("undeclared sym {}", name)
      }
    }
  }
}

pub fn translate_lvalue_expr(state: &mut OutputState, expr: &hir::Expr) -> Word {
  match expr.kind {
    hir::ExprKind::Variable(s) => {
      emit_sym(state, s)
    }
    _ => panic!()
  }
}

pub fn translate_const(state: &mut OutputState, c: &hir::Expr) -> Word {
  match c.kind {

    _ => panic!(),
  }
}

pub fn translate_vec4(state: &mut OutputState, x: &hir::Expr, y: &hir::Expr, z: &hir::Expr, w: &hir::Expr) -> Word {
  let float_vec4 = emit_vec4(state);
  let args = [
    translate_r_val(state, x),
    translate_r_val(state, y),
    translate_r_val(state, w),
    translate_r_val(state, z)];

  state.builder.composite_construct(float_vec4, None, args).unwrap()
}

pub fn translate_r_val(state: &mut OutputState, expr: &hir::Expr) -> Word {
  match expr.kind {
    hir::ExprKind::FunCall(ref fun, ref args) => {
      match fun {
        hir::FunIdentifier::Identifier(ref sym) => {
          let name = state.hir.sym(*sym).name.as_str();
            match name {
              "vec4" => {
                match args[..] {
                  [ref x, ref y, ref w, ref z] => translate_vec4(state, x, y, w, z),
                  [ref x] => translate_vec4(state, x, x, x, x),
                  _ => panic!(),
                }
              }
              _ => { panic!() }
            }
          }
        _ => panic!(),
      }
    }
    hir::ExprKind::DoubleConst(f) => {
      // XXX: we need to do something better about the types of literals
      // We could constant pool these things
      let float = emit_float(state);
      let b = &mut state.builder;
      b.constant_f32(float, f as f32)
    }
    hir::ExprKind::Variable(sym) => {
      let v = &state.emitted_syms[&sym];
      state.builder.load(v.ty, None, v.location, None, []).unwrap()
    }
    hir::ExprKind::Binary(ref op, ref l, ref r) => {
      let l = translate_r_val(state, l);
      let r = translate_r_val(state, r);
      match op {
        syntax::BinaryOp::Add => {
          let ty = emit_type(state, &expr.ty);
          state.builder.fadd(ty, None, l, r).unwrap()
        }
        _ => panic!("Unhandled op {:?}", op)
      }
    }
    _ => panic!("Unhandled {:?}", expr)
  }
}

pub fn translate_hir_expr(state: &mut OutputState, expr: &hir::Expr) {
  match expr.kind {

    hir::ExprKind::Assignment(ref v, ref op, ref e) => {
      let output_var = translate_lvalue_expr(state, v);
      let result = translate_r_val(state, e);
      let _ = state.builder.store(output_var, result, None, []);
    }
    _ => {}
  }
}

pub fn show_hir_expr<F>(f: &mut F, state: &mut OutputState, expr: &hir::Expr) where F: Write {
  match expr.kind {
    hir::ExprKind::Variable(ref i) => show_sym(f, state, i),
    hir::ExprKind::IntConst(ref x) => { let _ = write!(f, "{}", x); }
    hir::ExprKind::UIntConst(ref x) => { let _ = write!(f, "{}u", x); }
    hir::ExprKind::BoolConst(ref x) => { let _ = write!(f, "{}", x); }
    hir::ExprKind::FloatConst(ref x) => show_float(f, *x),
    hir::ExprKind::DoubleConst(ref x) => show_double(f, *x),
    hir::ExprKind::Unary(ref op, ref e) => {
      show_unary_op(f, &op);
      let _ = f.write_str("(");
      show_hir_expr(f, state, &e);
      let _ = f.write_str(")");
    }
    hir::ExprKind::Binary(ref op, ref l, ref r) => {
      let _ = f.write_str("(");
      show_hir_expr(f, state, &l);
      let _ = f.write_str(")");
      show_binary_op(f, &op);
      let _ = f.write_str("(");
      show_hir_expr(f, state, &r);
      let _ = f.write_str(")");
    }
    hir::ExprKind::Ternary(ref c, ref s, ref e) => {
      show_hir_expr(f, state, &c);
      let _ = f.write_str(" ? ");
      show_hir_expr(f, state, &s);
      let _ = f.write_str(" : ");
      show_hir_expr(f, state, &e);
    }
    hir::ExprKind::Assignment(ref v, ref op, ref e) => {
      show_hir_expr(f, state, &v);
      let _ = f.write_str(" ");
      show_assignment_op(f, &op);
      let _ = f.write_str(" ");
      show_hir_expr(f, state, &e);
    }
    hir::ExprKind::Bracket(ref e, ref indx) => {
      show_hir_expr(f, state, &e);
      let _ = f.write_str("[");
      show_hir_expr(f, state, &indx);
      let _ = f.write_str("]");
    }
    hir::ExprKind::FunCall(ref fun, ref args) => {
      show_hir_function_identifier(f, state, &fun);
      let _ = f.write_str("(");

      if !args.is_empty() {
        let mut args_iter = args.iter();
        let first = args_iter.next().unwrap();
        show_hir_expr(f, state, first);

        for e in args_iter {
          let _ = f.write_str(", ");
          show_hir_expr(f, state, e);
        }
      }

      let _ = f.write_str(")");
    }
    hir::ExprKind::Dot(ref e, ref i) => {
      let _ = f.write_str("(");
      show_hir_expr(f, state, &e);
      let _ = f.write_str(")");
      let _ = f.write_str(".");
      show_identifier(f, i);
    }
    hir::ExprKind::SwizzleSelector(ref e, ref s) => {
      let _ = f.write_str("(");
      show_hir_expr(f, state, &e);
      let _ = f.write_str(")");
      let _ = f.write_str(".");
      let _ = f.write_str(&s.to_string());
    }
    hir::ExprKind::PostInc(ref e) => {
      show_hir_expr(f, state, &e);
      let _ = f.write_str("++");
    }
    hir::ExprKind::PostDec(ref e) => {
      show_hir_expr(f, state, &e);
      let _ = f.write_str("--");
    }
    hir::ExprKind::Comma(ref a, ref b) => {
      show_hir_expr(f, state, &a);
      let _ = f.write_str(", ");
      show_hir_expr(f, state, &b);
    }
  }
}

pub fn show_expr<F>(f: &mut F, expr: &syntax::Expr) where F: Write {
  match *expr {
    syntax::Expr::Variable(ref i) => show_identifier(f, &i),
    syntax::Expr::IntConst(ref x) => { let _ = write!(f, "{}", x); }
    syntax::Expr::UIntConst(ref x) => { let _ = write!(f, "{}u", x); }
    syntax::Expr::BoolConst(ref x) => { let _ = write!(f, "{}", x); }
    syntax::Expr::FloatConst(ref x) => show_float(f, *x),
    syntax::Expr::DoubleConst(ref x) => show_double(f, *x),
    syntax::Expr::Unary(ref op, ref e) => {
      show_unary_op(f, &op);
      let _ = f.write_str("(");
      show_expr(f, &e);
      let _ = f.write_str(")");
    }
    syntax::Expr::Binary(ref op, ref l, ref r) => {
      let _ = f.write_str("(");
      show_expr(f, &l);
      let _ = f.write_str(")");
      show_binary_op(f, &op);
      let _ = f.write_str("(");
      show_expr(f, &r);
      let _ = f.write_str(")");
    }
    syntax::Expr::Ternary(ref c, ref s, ref e) => {
      show_expr(f, &c);
      let _ = f.write_str(" ? ");
      show_expr(f, &s);
      let _ = f.write_str(" : ");
      show_expr(f, &e);
    }
    syntax::Expr::Assignment(ref v, ref op, ref e) => {
      show_expr(f, &v);
      let _ = f.write_str(" ");
      show_assignment_op(f, &op);
      let _ = f.write_str(" ");
      show_expr(f, &e);
    }
    syntax::Expr::Bracket(ref e, ref a) => {
      show_expr(f, &e);
      //show_array_spec(f, &a);
    }
    syntax::Expr::FunCall(ref fun, ref args) => {
      show_function_identifier(f, &fun);
      let _ = f.write_str("(");

      if !args.is_empty() {
        let mut args_iter = args.iter();
        let first = args_iter.next().unwrap();
        show_expr(f, first);

        for e in args_iter {
          let _ = f.write_str(", ");
          show_expr(f, e);
        }
      }

      let _ = f.write_str(")");
    }
    syntax::Expr::Dot(ref e, ref i) => {
      let _ = f.write_str("(");
      show_expr(f, &e);
      let _ = f.write_str(")");
      let _ = f.write_str(".");
      show_identifier(f, &i);
    }
    syntax::Expr::PostInc(ref e) => {
      show_expr(f, &e);
      let _ = f.write_str("++");
    }
    syntax::Expr::PostDec(ref e) => {
      show_expr(f, &e);
      let _ = f.write_str("--");
    }
    syntax::Expr::Comma(ref a, ref b) => {
      show_expr(f, &a);
      let _ = f.write_str(", ");
      show_expr(f, &b);
    }
  }
}

pub fn show_unary_op<F>(f: &mut F, op: &syntax::UnaryOp) where F: Write {
  match *op {
    syntax::UnaryOp::Inc => { let _ = f.write_str("++"); }
    syntax::UnaryOp::Dec => { let _ = f.write_str("--"); }
    syntax::UnaryOp::Add => { let _ = f.write_str("+"); }
    syntax::UnaryOp::Minus => { let _ = f.write_str("-"); }
    syntax::UnaryOp::Not => { let _ = f.write_str("!"); }
    syntax::UnaryOp::Complement => { let _ = f.write_str("~"); }
  }
}

pub fn show_binary_op<F>(f: &mut F, op: &syntax::BinaryOp) where F: Write {
  match *op {
    syntax::BinaryOp::Or => { let _ = f.write_str("||"); }
    syntax::BinaryOp::Xor => { let _ = f.write_str("^^"); }
    syntax::BinaryOp::And => { let _ = f.write_str("&&"); }
    syntax::BinaryOp::BitOr => { let _ = f.write_str("|"); }
    syntax::BinaryOp::BitXor => { let _ = f.write_str("^"); }
    syntax::BinaryOp::BitAnd => { let _ = f.write_str("&"); }
    syntax::BinaryOp::Equal => { let _ = f.write_str("=="); }
    syntax::BinaryOp::NonEqual => { let _ = f.write_str("!="); }
    syntax::BinaryOp::LT => { let _ = f.write_str("<"); }
    syntax::BinaryOp::GT => { let _ = f.write_str(">"); }
    syntax::BinaryOp::LTE => { let _ = f.write_str("<="); }
    syntax::BinaryOp::GTE => { let _ = f.write_str(">="); }
    syntax::BinaryOp::LShift => { let _ = f.write_str("<<"); }
    syntax::BinaryOp::RShift => { let _ = f.write_str(">>"); }
    syntax::BinaryOp::Add => { let _ = f.write_str("+"); }
    syntax::BinaryOp::Sub => { let _ = f.write_str("-"); }
    syntax::BinaryOp::Mult => { let _ = f.write_str("*"); }
    syntax::BinaryOp::Div => { let _ = f.write_str("/"); }
    syntax::BinaryOp::Mod => { let _ = f.write_str("%"); }
  }
}

pub fn show_assignment_op<F>(f: &mut F, op: &syntax::AssignmentOp) where F: Write {
  match *op {
    syntax::AssignmentOp::Equal => { let _ = f.write_str("="); }
    syntax::AssignmentOp::Mult => { let _ = f.write_str("*="); }
    syntax::AssignmentOp::Div => { let _ = f.write_str("/="); }
    syntax::AssignmentOp::Mod => { let _ = f.write_str("%="); }
    syntax::AssignmentOp::Add => { let _ = f.write_str("+="); }
    syntax::AssignmentOp::Sub => { let _ = f.write_str("-="); }
    syntax::AssignmentOp::LShift => { let _ = f.write_str("<<="); }
    syntax::AssignmentOp::RShift => { let _ = f.write_str(">>="); }
    syntax::AssignmentOp::And => { let _ = f.write_str("&="); }
    syntax::AssignmentOp::Xor => { let _ = f.write_str("^="); }
    syntax::AssignmentOp::Or => { let _ = f.write_str("|="); }
  }
}

pub fn show_function_identifier<F>(f: &mut F, i: &syntax::FunIdentifier) where F: Write {
  match *i {
    syntax::FunIdentifier::Identifier(ref n) => show_identifier(f, &n),
    syntax::FunIdentifier::Expr(ref e) => show_expr(f, &*e)
  }
}

pub fn show_hir_function_identifier<F>(f: &mut F, state: &mut OutputState, i: &hir::FunIdentifier) where F: Write {
  match *i {
    hir::FunIdentifier::Identifier(ref n) => show_sym(f, state, n),
    hir::FunIdentifier::Constructor(ref t) => show_type(f, state, t)
  }
}

pub fn translate_declaration(state: &mut OutputState, d: &hir::Declaration) {
  match *d {
    hir::Declaration::InitDeclaratorList(ref list) => {

      translate_init_declarator_list(state, &list);
    }
    _ => panic!()
  }
}

pub fn show_declaration<F>(f: &mut F, state: &mut OutputState, d: &hir::Declaration) where F: Write {
  match *d {
    hir::Declaration::FunctionPrototype(ref proto) => {
      show_function_prototype(f, state, &proto);
      let _ = f.write_str(";\n");
    }
    hir::Declaration::InitDeclaratorList(ref list) => {
      show_init_declarator_list(f, state, &list);
      let _ = f.write_str(";\n");
    }
    hir::Declaration::Precision(ref qual, ref ty) => {
      show_precision_qualifier(f, &qual);
      show_type_specifier(f, state, &ty);
      let _ = f.write_str(";\n");
    }
    hir::Declaration::Block(ref block) => {
      show_block(f, state, &block);
      let _ = f.write_str(";\n");
    }
    hir::Declaration::Global(ref qual, ref identifiers) => {
      show_type_qualifier(f, &qual);

      if !identifiers.is_empty() {
        let mut iter = identifiers.iter();
        let first = iter.next().unwrap();
        show_identifier(f, first);

        for identifier in iter {
          let _ = write!(f, ", {}", identifier);
        }
      }

      let _ = f.write_str(";\n");
    }
    hir::Declaration::StructDefinition(sym) => {
      panic!()
    }
  }
}

pub fn show_function_prototype<F>(f: &mut F, state: &mut OutputState, fp: &hir::FunctionPrototype) where F: Write {
  show_type(f, state, &fp.ty);
  let _ = f.write_str(" ");
  show_identifier(f, &fp.name);

  let _ = f.write_str("(");

  if !fp.parameters.is_empty() {
    let mut iter = fp.parameters.iter();
    let first = iter.next().unwrap();
    show_function_parameter_declaration(f, state, first);

    for param in iter {
      let _ = f.write_str(", ");
      show_function_parameter_declaration(f, state, param);
    }
  }

  let _ = f.write_str(")");
}
pub fn show_function_parameter_declaration<F>(f: &mut F, state: &mut OutputState, p: &hir::FunctionParameterDeclaration) where F: Write {
  match *p {
    hir::FunctionParameterDeclaration::Named(ref qual, ref fpd) => {
      if let Some(ref q) = *qual {
        show_type_qualifier(f, q);
        let _ = f.write_str(" ");
      }

      show_function_parameter_declarator(f, state, fpd);
    }
    hir::FunctionParameterDeclaration::Unnamed(ref qual, ref ty) => {
      if let Some(ref q) = *qual {
        show_type_qualifier(f, q);
        let _ = f.write_str(" ");
      }

      show_type_specifier(f, state, ty);
    }
  }
}

pub fn show_function_parameter_declarator<F>(f: &mut F, state: &mut OutputState, p: &hir::FunctionParameterDeclarator) where F: Write {
  show_type(f, state, &p.ty);
  let _ = f.write_str(" ");
  show_arrayed_identifier(f, &p.ident, &p.ty);
}

pub fn show_init_declarator_list<F>(f: &mut F, state: &mut OutputState, i: &hir::InitDeclaratorList) where F: Write {
  show_single_declaration(f, state, &i.head);

  for decl in &i.tail {
    let _ = f.write_str(", ");
    show_single_declaration_no_type(f, state, decl);
  }
}

pub fn translate_initializer(state: &mut OutputState, i: &hir::Initializer) -> Word {
  match *i {
    hir::Initializer::Simple(ref e) => translate_r_val(state, e),
    _ => panic!(),
  }
}

pub fn translate_single_declaration(state: &mut OutputState, d: &hir::SingleDeclaration) {

  let ty = emit_type(state, &d.ty);

  let storage = match &state.hir.sym(d.name).decl {
    hir::SymDecl::Variable(storage, _) => {
      match storage {
        hir::StorageClass::Const => spirv::StorageClass::UniformConstant,
        hir::StorageClass::Out => spirv::StorageClass::Output,
        hir::StorageClass::In => spirv::StorageClass::Input,
        hir::StorageClass::Uniform => spirv::StorageClass::Uniform,
        hir::StorageClass::None => spirv::StorageClass::Function
      }
    }
    _ => panic!(),
  };

  let output_var = state.builder.variable(ty, None, storage, None);
  state.emitted_syms.insert(d.name,  Variable{ location: output_var, ty });

  if let Some(ref initializer) = d.initializer {
    let init_val = translate_initializer(state, initializer);
    state.builder.store(output_var, init_val, None, []);
  }
}

pub fn translate_init_declarator_list(state: &mut OutputState, i: &hir::InitDeclaratorList) {
  translate_single_declaration(state, &i.head);

  for decl in &i.tail {
    panic!()
  }
}

pub fn show_single_declaration<F>(f: &mut F, state: &mut OutputState, d: &hir::SingleDeclaration) where F: Write {
  //show_fully_specified_type(f, state, &d.ty);

  let _ = f.write_str(" ");
  show_sym(f, state, &d.name);

  /*if let Some(ref arr_spec) = d.array_specifier {
    show_array_spec(f, arr_spec);
  }*/

  if let Some(ref initializer) = d.initializer {
    let _ = f.write_str(" = ");
    show_initializer(f, state, initializer);
  }
}

pub fn show_single_declaration_no_type<F>(f: &mut F, state: &mut OutputState, d: &hir::SingleDeclarationNoType) where F: Write {
  /*show_arrayed_identifier(f, &d.ident);

  if let Some(ref initializer) = d.initializer {
    let _ = f.write_str(" = ");
    show_initializer(f, state, initializer);
  }*/
}

pub fn show_initializer<F>(f: &mut F, state: &mut OutputState, i: &hir::Initializer) where F: Write {
  match *i {
    hir::Initializer::Simple(ref e) => show_hir_expr(f, state, e),
    hir::Initializer::List(ref list) => {
      let mut iter = list.0.iter();
      let first = iter.next().unwrap();

      let _ = f.write_str("{ ");
      show_initializer(f, state, first);

      for ini in iter {
        let _ = f.write_str(", ");
        show_initializer(f, state, ini);
      }

      let _ = f.write_str(" }");
    }
  }
}

pub fn show_block<F>(f: &mut F, state: &mut OutputState, b: &hir::Block) where F: Write {
  show_type_qualifier(f, &b.qualifier);
  let _ = f.write_str(" ");
  show_identifier(f, &b.name);
  let _ = f.write_str(" {");

  for field in &b.fields {
    show_struct_field(f, state, field);
    let _ = f.write_str("\n");
  }
  let _ = f.write_str("}");

  if let Some(ref ident) = b.identifier {
    //show_arrayed_identifier(f, ident);
  }
}

pub fn translate_type(state: &mut OutputState, ty: &hir::Type) -> spirv::Word {
  emit_void(state)
}

pub fn translate_function_definition(state: &mut OutputState, fd: &hir::FunctionDefinition) {
  state.return_type = Some(Box::new(fd.prototype.ty.clone()));


  let ret_type = translate_type(state, &fd.prototype.ty);
  {
    let void = emit_void(state);
    let b = &mut state.builder;
    let voidf = b.type_function(void, vec![]);

    let fun = b.begin_function(ret_type,
                     None,
                     (spirv::FunctionControl::DONT_INLINE |
                         spirv::FunctionControl::CONST),
                     voidf)
        .unwrap();
    b.execution_mode(fun, spirv::ExecutionMode::OriginUpperLeft, []);
    b.entry_point(state.model, fun, "main", []);

    b.begin_basic_block(None).unwrap();
  }
  translate_compound_statement(state, &fd.statement);

  let b = &mut state.builder;
  b.ret().unwrap();
  b.end_function().unwrap();
  state.return_type = None;
  state.return_declared = false;
}

pub fn show_compound_statement<F>(f: &mut F, state: &mut OutputState, cst: &hir::CompoundStatement) where F: Write {
  let _ = f.write_str("{\n");

  for st in &cst.statement_list {
    show_statement(f, state, st);
  }

  let _ = f.write_str("}\n");
}

pub fn translate_compound_statement(state: &mut OutputState, cst: &hir::CompoundStatement) {
  for st in &cst.statement_list {
    translate_statement(state, st);
  }
}


pub fn translate_statement(state: &mut OutputState, st: &hir::Statement) {
  match *st {
    hir::Statement::Compound(ref cst) => translate_compound_statement(state, cst),
    hir::Statement::Simple(ref sst) => translate_simple_statement(state, sst)
  }
}

pub fn show_statement<F>(f: &mut F, state: &mut OutputState, st: &hir::Statement) where F: Write {
  match *st {
    hir::Statement::Compound(ref cst) => show_compound_statement(f, state, cst),
    hir::Statement::Simple(ref sst) => show_simple_statement(f, state, sst)
  }
}

pub fn show_simple_statement<F>(f: &mut F, state: &mut OutputState, sst: &hir::SimpleStatement) where F: Write {
  match *sst {
    hir::SimpleStatement::Declaration(ref d) => show_declaration(f, state, d),
    hir::SimpleStatement::Expression(ref e) => show_expression_statement(f, state, e),
    hir::SimpleStatement::Selection(ref s) => show_selection_statement(f, state, s),
    hir::SimpleStatement::Switch(ref s) => show_switch_statement(f, state, s),
    hir::SimpleStatement::Iteration(ref i) => show_iteration_statement(f, state, i),
    hir::SimpleStatement::Jump(ref j) => show_jump_statement(f, state, j)
  }
}

pub fn translate_simple_statement(state: &mut OutputState, sst: &hir::SimpleStatement) {
  match *sst {
    hir::SimpleStatement::Declaration(ref d) => translate_declaration(state, d),
    hir::SimpleStatement::Expression(ref e) => translate_expression_statement(state, e),
    hir::SimpleStatement::Selection(ref s) => panic!(), //show_selection_statement(f, state, s),
    hir::SimpleStatement::Switch(ref s) => panic!(), //show_switch_statement(f, state, s),
    hir::SimpleStatement::Iteration(ref i) => panic!(), //show_iteration_statement(f, state, i),
    hir::SimpleStatement::Jump(ref j) => panic!(), //show_jump_statement(f, state, j)
  }
}

pub fn show_expression_statement<F>(f: &mut F, state: &mut OutputState, est: &hir::ExprStatement) where F: Write {
  if let Some(ref e) = *est {
    show_hir_expr(f, state, e);
  }

  let _ = f.write_str(";\n");
}


pub fn translate_expression_statement(state: &mut OutputState, est: &hir::ExprStatement) {
  if let Some(ref e) = *est {
    translate_hir_expr(state, e);
  }
}

pub fn show_selection_statement<F>(f: &mut F, state: &mut OutputState, sst: &hir::SelectionStatement) where F: Write {
    let _ = f.write_str("if (");
    show_hir_expr(f, state, &sst.cond);
    let _ = f.write_str(") {\n");
    show_selection_rest_statement(f, state, &sst.rest);
}

pub fn show_selection_rest_statement<F>(f: &mut F, state: &mut OutputState, sst: &hir::SelectionRestStatement) where F: Write {
  match *sst {
    hir::SelectionRestStatement::Statement(ref if_st) => {
      show_statement(f, state, if_st);
      let _ = f.write_str("}\n");
    }
    hir::SelectionRestStatement::Else(ref if_st, ref else_st) => {
      show_statement(f, state, if_st);
      let _ = f.write_str("} else ");
      show_statement(f, state, else_st);
    }
  }
}

pub fn show_switch_statement<F>(f: &mut F, state: &mut OutputState, sst: &hir::SwitchStatement) where F: Write {
  let _ = f.write_str("switch (");
  show_hir_expr(f, state, &sst.head);
  let _ = f.write_str(") {\n");

  for case in &sst.cases {
    show_case_label(f, state, &case.label);
    for st in &case.stmts {
      show_statement(f, state, st);
    }
  }
  let _ = f.write_str("}\n");

}

pub fn show_case_label<F>(f: &mut F, state: &mut OutputState, cl: &hir::CaseLabel) where F: Write {
  match *cl {
    hir::CaseLabel::Case(ref e) => {
      let _ = f.write_str("case ");
      show_hir_expr(f, state, e);
      let _ = f.write_str(":\n");
    }
    hir::CaseLabel::Def => { let _ = f.write_str("default:\n"); }
  }
}

pub fn show_iteration_statement<F>(f: &mut F, state: &mut OutputState, ist: &hir::IterationStatement) where F: Write {
  match *ist {
    hir::IterationStatement::While(ref cond, ref body) => {
      let _ = f.write_str("while (");
      show_condition(f, state, cond);
      let _ = f.write_str(") ");
      show_statement(f, state, body);
    }
    hir::IterationStatement::DoWhile(ref body, ref cond) => {
      let _ = f.write_str("do ");
      show_statement(f, state, body);
      let _ = f.write_str(" while (");
      show_hir_expr(f, state, cond);
      let _ = f.write_str(")\n");
    }
    hir::IterationStatement::For(ref init, ref rest, ref body) => {
      let _ = f.write_str("for (");
      show_for_init_statement(f, state, init);
      show_for_rest_statement(f, state, rest);
      let _ = f.write_str(") ");
      show_statement(f, state, body);
    }
  }
}

pub fn show_condition<F>(f: &mut F, state: &mut OutputState, c: &hir::Condition) where F: Write {
  match *c {
    hir::Condition::Expr(ref e) => show_hir_expr(f, state, e),
    /*hir::Condition::Assignment(ref ty, ref name, ref initializer) => {
      show_type(f, state, ty);
      let _ = f.write_str(" ");
      show_identifier(f, name);
      let _ = f.write_str(" = ");
      show_initializer(f, state, initializer);
    }*/
  }
}

pub fn show_for_init_statement<F>(f: &mut F, state: &mut OutputState, i: &hir::ForInitStatement) where F: Write {
  match *i {
    hir::ForInitStatement::Expression(ref expr) => {
      if let Some(ref e) = *expr {
        show_hir_expr(f, state, e);
      }
    }
    hir::ForInitStatement::Declaration(ref d) => {
      show_declaration(f, state, d);
    }
  }
}

pub fn show_for_rest_statement<F>(f: &mut F, state: &mut OutputState, r: &hir::ForRestStatement) where F: Write {
  if let Some(ref cond) = r.condition {
    show_condition(f, state, cond);
  }

  let _ = f.write_str("; ");

  if let Some(ref e) = r.post_expr {
    show_hir_expr(f, state, e);
  }
}

pub fn show_jump_statement<F>(f: &mut F, state: &mut OutputState, j: &hir::JumpStatement) where F: Write {
  match *j {
    hir::JumpStatement::Continue => { let _ = f.write_str("continue;\n"); }
    hir::JumpStatement::Break => { let _ = f.write_str("break;\n"); }
    hir::JumpStatement::Discard => { let _ = f.write_str("discard;\n"); }
    hir::JumpStatement::Return(ref e) => {
      let _ = f.write_str("return ");
      show_hir_expr(f, state, e);
      let _ = f.write_str(";\n");
    }
  }
}

pub fn translate_external_declaration(state: &mut OutputState, ed: &hir::ExternalDeclaration) {
  match *ed {
    hir::ExternalDeclaration::Preprocessor(ref pp) => panic!("Preprocessor unsupported"),
    hir::ExternalDeclaration::FunctionDefinition(ref fd) => translate_function_definition(state, fd),
    hir::ExternalDeclaration::Declaration(ref d) => translate_declaration(state, d),
  }
}

pub fn translate_translation_unit(state: &mut OutputState, tu: &hir::TranslationUnit) {
  for ed in &(tu.0).0 {
    translate_external_declaration(state, ed);
  }
}

#[test]
fn basic() {
  let s= glsl_to_spirv("

void main()
{
    float x = 0.1;
    x = x + 0.1;
	gl_FragColor = vec4(x, 0.4, 0.8, 1.0);
}");
  assert_eq!(s, r#"; SPIR-V
; Version: 1.3
; Generator: rspirv
; Bound: 19
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %3 "main"
OpExecutionMode %3 OriginUpperLeft
OpDecorate %13 Location 0
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%5 = OpTypeFloat 32
%7 = OpConstant  %5  0.1
%9 = OpConstant  %5  0.1
%11 = OpTypeVector %5 4
%12 = OpTypePointer Output %11
%15 = OpConstant  %5  0.4
%16 = OpConstant  %5  1.0
%17 = OpConstant  %5  0.8
%3 = OpFunction  %1  DontInline|Const %2
%4 = OpLabel
%6 = OpVariable  %5  Function
OpStore %6 %7
%8 = OpLoad  %5  %6
%10 = OpFAdd  %5  %8 %9
OpStore %6 %10
%13 = OpVariable  %12  Output
%14 = OpLoad  %5  %6
%18 = OpCompositeConstruct  %11  %14 %15 %16 %17
OpStore %13 %18
OpReturn
OpFunctionEnd"#)
}


#[test]
fn vec_addition() {
  let s= glsl_to_spirv("void main() {
    vec4 p = vec4(1.);
	gl_FragColor = vec4(0.4, 0.4, 0.8, 1.0) + p;
}");
  let reference = r#"; SPIR-V
; Version: 1.3
; Generator: rspirv
; Bound: 22
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %3 "main"
OpExecutionMode %3 OriginUpperLeft
OpDecorate %14 Location 0
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%5 = OpTypeFloat 32
%6 = OpTypeVector %5 4
%8 = OpConstant  %5  1.0
%9 = OpConstant  %5  1.0
%10 = OpConstant  %5  1.0
%11 = OpConstant  %5  1.0
%13 = OpTypePointer Output %6
%15 = OpConstant  %5  0.4
%16 = OpConstant  %5  0.4
%17 = OpConstant  %5  1.0
%18 = OpConstant  %5  0.8
%3 = OpFunction  %1  DontInline|Const %2
%4 = OpLabel
%7 = OpVariable  %6  Function
%12 = OpCompositeConstruct  %6  %8 %9 %10 %11
OpStore %7 %12
%14 = OpVariable  %13  Output
%19 = OpCompositeConstruct  %6  %15 %16 %17 %18
%20 = OpLoad  %6  %7
%21 = OpFAdd  %6  %19 %20
OpStore %14 %21
OpReturn
OpFunctionEnd"#;

  if s != reference {
    println!("{}", s);
    println!("{}", reference);
    panic!()
  }
}
