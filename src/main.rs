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
use std::collections::hash_map::Entry;

mod hir;

use hir::State;

fn glsl_to_spirv(input: &str) -> String {
  let ast = TranslationUnit::parse(input).unwrap();
  let mut state = hir::State::new();
  let hir = hir::ast_to_hir(&mut state, &ast);
  let mut b = rspirv::mr::Builder::new();
  b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);


  let mut state = OutputState { hir: state,
    return_type: None,
    return_declared: false,
    builder: b,
    emitted_types: HashMap::new(),
    emitted_syms: HashMap::new(),
  };

  translate_translation_unit(&mut state, &hir);

  let mut b = state.builder;

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
  let r = TranslationUnit::parse("

void main()
{
    vec4 p = vec4(1.);
	gl_FragColor = vec4(0.4, 0.4, 0.8, 1.0) + p;
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
  b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);


  let mut state = OutputState { hir: state,
    return_type: None,
    return_declared: false,
    builder: b,
    emitted_types: HashMap::new(),
    emitted_syms: HashMap::new(),
  };

  translate_translation_unit(&mut state, &hir);

  let mut b = state.builder;

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
  println!("{}", module.disassemble());
}

pub struct Variable {
  location: Word,
  ty: Word,
}

pub struct OutputState {
  builder: rspirv::mr::Builder,
  hir: hir::State,
  return_type: Option<Box<syntax::FullySpecifiedType>>,
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

pub fn show_type_specifier_non_array<F>(f: &mut F, state: &mut OutputState, t: &syntax::TypeSpecifierNonArray) where F: Write {
  match *t {
    syntax::TypeSpecifierNonArray::Void => { let _ = f.write_str("void"); }
    syntax::TypeSpecifierNonArray::Bool => { let _ = f.write_str("bool"); }
    syntax::TypeSpecifierNonArray::Int => { let _ = f.write_str("int"); }
    syntax::TypeSpecifierNonArray::UInt => { let _ = f.write_str("uint"); }
    syntax::TypeSpecifierNonArray::Float => { let _ = f.write_str("float"); }
    syntax::TypeSpecifierNonArray::Double => { let _ = f.write_str("double"); }
    syntax::TypeSpecifierNonArray::Vec2 => { let _ = f.write_str("vec2"); }
    syntax::TypeSpecifierNonArray::Vec3 => { let _ = f.write_str("vec3"); }
    syntax::TypeSpecifierNonArray::Vec4 => { let _ = f.write_str("vec4"); }
    syntax::TypeSpecifierNonArray::DVec2 => { let _ = f.write_str("dvec2"); }
    syntax::TypeSpecifierNonArray::DVec3 => { let _ = f.write_str("dvec3"); }
    syntax::TypeSpecifierNonArray::DVec4 => { let _ = f.write_str("dvec4"); }
    syntax::TypeSpecifierNonArray::BVec2 => { let _ = f.write_str("bvec2"); }
    syntax::TypeSpecifierNonArray::BVec3 => { let _ = f.write_str("bvec3"); }
    syntax::TypeSpecifierNonArray::BVec4 => { let _ = f.write_str("bvec4"); }
    syntax::TypeSpecifierNonArray::IVec2 => { let _ = f.write_str("ivec2"); }
    syntax::TypeSpecifierNonArray::IVec3 => { let _ = f.write_str("ivec3"); }
    syntax::TypeSpecifierNonArray::IVec4 => { let _ = f.write_str("ivec4"); }
    syntax::TypeSpecifierNonArray::UVec2 => { let _ = f.write_str("uvec2"); }
    syntax::TypeSpecifierNonArray::UVec3 => { let _ = f.write_str("uvec3"); }
    syntax::TypeSpecifierNonArray::UVec4 => { let _ = f.write_str("uvec4"); }
    syntax::TypeSpecifierNonArray::Mat2 => { let _ = f.write_str("mat2"); }
    syntax::TypeSpecifierNonArray::Mat3 => { let _ = f.write_str("mat3"); }
    syntax::TypeSpecifierNonArray::Mat4 => { let _ = f.write_str("mat4"); }
    syntax::TypeSpecifierNonArray::Mat23 => { let _ = f.write_str("mat23"); }
    syntax::TypeSpecifierNonArray::Mat24 => { let _ = f.write_str("mat24"); }
    syntax::TypeSpecifierNonArray::Mat32 => { let _ = f.write_str("mat32"); }
    syntax::TypeSpecifierNonArray::Mat34 => { let _ = f.write_str("mat34"); }
    syntax::TypeSpecifierNonArray::Mat42 => { let _ = f.write_str("mat42"); }
    syntax::TypeSpecifierNonArray::Mat43 => { let _ = f.write_str("mat43"); }
    syntax::TypeSpecifierNonArray::DMat2 => { let _ = f.write_str("dmat2"); }
    syntax::TypeSpecifierNonArray::DMat3 => { let _ = f.write_str("dmat3"); }
    syntax::TypeSpecifierNonArray::DMat4 => { let _ = f.write_str("dmat4"); }
    syntax::TypeSpecifierNonArray::DMat23 => { let _ = f.write_str("dmat23"); }
    syntax::TypeSpecifierNonArray::DMat24 => { let _ = f.write_str("dmat24"); }
    syntax::TypeSpecifierNonArray::DMat32 => { let _ = f.write_str("dmat32"); }
    syntax::TypeSpecifierNonArray::DMat34 => { let _ = f.write_str("dmat34"); }
    syntax::TypeSpecifierNonArray::DMat42 => { let _ = f.write_str("dmat42"); }
    syntax::TypeSpecifierNonArray::DMat43 => { let _ = f.write_str("dmat43"); }
    syntax::TypeSpecifierNonArray::Sampler1D => { let _ = f.write_str("sampler1D"); }
    syntax::TypeSpecifierNonArray::Image1D => { let _ = f.write_str("image1D"); }
    syntax::TypeSpecifierNonArray::Sampler2D => { let _ = f.write_str("sampler2D"); }
    syntax::TypeSpecifierNonArray::Image2D => { let _ = f.write_str("image2D"); }
    syntax::TypeSpecifierNonArray::Sampler3D => { let _ = f.write_str("sampler3D"); }
    syntax::TypeSpecifierNonArray::Image3D => { let _ = f.write_str("image3D"); }
    syntax::TypeSpecifierNonArray::SamplerCube => { let _ = f.write_str("samplerCube"); }
    syntax::TypeSpecifierNonArray::ImageCube => { let _ = f.write_str("imageCube"); }
    syntax::TypeSpecifierNonArray::Sampler2DRect => { let _ = f.write_str("sampler2DRect"); }
    syntax::TypeSpecifierNonArray::Image2DRect => { let _ = f.write_str("image2DRect"); }
    syntax::TypeSpecifierNonArray::Sampler1DArray => { let _ = f.write_str("sampler1DArray"); }
    syntax::TypeSpecifierNonArray::Image1DArray => { let _ = f.write_str("image1DArray"); }
    syntax::TypeSpecifierNonArray::Sampler2DArray => { let _ = f.write_str("sampler2DArray"); }
    syntax::TypeSpecifierNonArray::Image2DArray => { let _ = f.write_str("image2DArray"); }
    syntax::TypeSpecifierNonArray::SamplerBuffer => { let _ = f.write_str("samplerBuffer"); }
    syntax::TypeSpecifierNonArray::ImageBuffer => { let _ = f.write_str("imageBuffer"); }
    syntax::TypeSpecifierNonArray::Sampler2DMS => { let _ = f.write_str("sampler2DMS"); }
    syntax::TypeSpecifierNonArray::Image2DMS => { let _ = f.write_str("image2DMS"); }
    syntax::TypeSpecifierNonArray::Sampler2DMSArray => { let _ = f.write_str("sampler2DMSArray"); }
    syntax::TypeSpecifierNonArray::Image2DMSArray => { let _ = f.write_str("image2DMSArray"); }
    syntax::TypeSpecifierNonArray::SamplerCubeArray => { let _ = f.write_str("samplerCubeArray"); }
    syntax::TypeSpecifierNonArray::ImageCubeArray => { let _ = f.write_str("imageCubeArray"); }
    syntax::TypeSpecifierNonArray::Sampler1DShadow => { let _ = f.write_str("sampler1DShadow"); }
    syntax::TypeSpecifierNonArray::Sampler2DShadow => { let _ = f.write_str("sampler2DShadow"); }
    syntax::TypeSpecifierNonArray::Sampler2DRectShadow => { let _ = f.write_str("sampler2DRectShadow"); }
    syntax::TypeSpecifierNonArray::Sampler1DArrayShadow => { let _ = f.write_str("sampler1DArrayShadow"); }
    syntax::TypeSpecifierNonArray::Sampler2DArrayShadow => { let _ = f.write_str("sampler2DArrayShadow"); }
    syntax::TypeSpecifierNonArray::SamplerCubeShadow => { let _ = f.write_str("samplerCubeShadow"); }
    syntax::TypeSpecifierNonArray::SamplerCubeArrayShadow => { let _ = f.write_str("samplerCubeArrayShadow"); }
    syntax::TypeSpecifierNonArray::ISampler1D => { let _ = f.write_str("isampler1D"); }
    syntax::TypeSpecifierNonArray::IImage1D => { let _ = f.write_str("iimage1D"); }
    syntax::TypeSpecifierNonArray::ISampler2D => { let _ = f.write_str("isampler2D"); }
    syntax::TypeSpecifierNonArray::IImage2D => { let _ = f.write_str("iimage2D"); }
    syntax::TypeSpecifierNonArray::ISampler3D => { let _ = f.write_str("isampler3D"); }
    syntax::TypeSpecifierNonArray::IImage3D => { let _ = f.write_str("iimage3D"); }
    syntax::TypeSpecifierNonArray::ISamplerCube => { let _ = f.write_str("isamplerCube"); }
    syntax::TypeSpecifierNonArray::IImageCube => { let _ = f.write_str("iimageCube"); }
    syntax::TypeSpecifierNonArray::ISampler2DRect => { let _ = f.write_str("isampler2DRect"); }
    syntax::TypeSpecifierNonArray::IImage2DRect => { let _ = f.write_str("iimage2DRect"); }
    syntax::TypeSpecifierNonArray::ISampler1DArray => { let _ = f.write_str("isampler1DArray"); }
    syntax::TypeSpecifierNonArray::IImage1DArray => { let _ = f.write_str("iimage1DArray"); }
    syntax::TypeSpecifierNonArray::ISampler2DArray => { let _ = f.write_str("isampler2DArray"); }
    syntax::TypeSpecifierNonArray::IImage2DArray => { let _ = f.write_str("iimage2DArray"); }
    syntax::TypeSpecifierNonArray::ISamplerBuffer => { let _ = f.write_str("isamplerBuffer"); }
    syntax::TypeSpecifierNonArray::IImageBuffer => { let _ = f.write_str("iimageBuffer"); }
    syntax::TypeSpecifierNonArray::ISampler2DMS => { let _ = f.write_str("isampler2MS"); }
    syntax::TypeSpecifierNonArray::IImage2DMS => { let _ = f.write_str("iimage2DMS"); }
    syntax::TypeSpecifierNonArray::ISampler2DMSArray => { let _ = f.write_str("isampler2DMSArray"); }
    syntax::TypeSpecifierNonArray::IImage2DMSArray => { let _ = f.write_str("iimage2DMSArray"); }
    syntax::TypeSpecifierNonArray::ISamplerCubeArray => { let _ = f.write_str("isamplerCubeArray"); }
    syntax::TypeSpecifierNonArray::IImageCubeArray => { let _ = f.write_str("iimageCubeArray"); }
    syntax::TypeSpecifierNonArray::AtomicUInt => { let _ = f.write_str("atomic_uint"); }
    syntax::TypeSpecifierNonArray::USampler1D => { let _ = f.write_str("usampler1D"); }
    syntax::TypeSpecifierNonArray::UImage1D => { let _ = f.write_str("uimage1D"); }
    syntax::TypeSpecifierNonArray::USampler2D => { let _ = f.write_str("usampler2D"); }
    syntax::TypeSpecifierNonArray::UImage2D => { let _ = f.write_str("uimage2D"); }
    syntax::TypeSpecifierNonArray::USampler3D => { let _ = f.write_str("usampler3D"); }
    syntax::TypeSpecifierNonArray::UImage3D => { let _ = f.write_str("uimage3D"); }
    syntax::TypeSpecifierNonArray::USamplerCube => { let _ = f.write_str("usamplerCube"); }
    syntax::TypeSpecifierNonArray::UImageCube => { let _ = f.write_str("uimageCube"); }
    syntax::TypeSpecifierNonArray::USampler2DRect => { let _ = f.write_str("usampler2DRect"); }
    syntax::TypeSpecifierNonArray::UImage2DRect => { let _ = f.write_str("uimage2DRect"); }
    syntax::TypeSpecifierNonArray::USampler1DArray => { let _ = f.write_str("usampler1DArray"); }
    syntax::TypeSpecifierNonArray::UImage1DArray => { let _ = f.write_str("uimage1DArray"); }
    syntax::TypeSpecifierNonArray::USampler2DArray => { let _ = f.write_str("usampler2DArray"); }
    syntax::TypeSpecifierNonArray::UImage2DArray => { let _ = f.write_str("uimage2DArray"); }
    syntax::TypeSpecifierNonArray::USamplerBuffer => { let _ = f.write_str("usamplerBuffer"); }
    syntax::TypeSpecifierNonArray::UImageBuffer => { let _ = f.write_str("uimageBuffer"); }
    syntax::TypeSpecifierNonArray::USampler2DMS => { let _ = f.write_str("usampler2DMS"); }
    syntax::TypeSpecifierNonArray::UImage2DMS => { let _ = f.write_str("uimage2DMS"); }
    syntax::TypeSpecifierNonArray::USampler2DMSArray => { let _ = f.write_str("usamplerDMSArray"); }
    syntax::TypeSpecifierNonArray::UImage2DMSArray => { let _ = f.write_str("uimage2DMSArray"); }
    syntax::TypeSpecifierNonArray::USamplerCubeArray => { let _ = f.write_str("usamplerCubeArray"); }
    syntax::TypeSpecifierNonArray::UImageCubeArray => { let _ = f.write_str("uimageCubeArray"); }
    syntax::TypeSpecifierNonArray::Struct(ref s) => show_struct_non_declaration(f, state, s),
    syntax::TypeSpecifierNonArray::TypeName(ref tn) => show_type_name(f, tn)
  }
}

pub fn show_type_specifier<F>(f: &mut F, state: &mut OutputState, t: &syntax::TypeSpecifier) where F: Write {
  show_type_specifier_non_array(f, state, &t.ty);

  if let Some(ref arr_spec) = t.array_specifier {
    show_array_spec(f, arr_spec);
  }
}

pub fn show_fully_specified_type<F>(f: &mut F, state: &mut OutputState, t: &syntax::FullySpecifiedType) where F: Write {
  if let Some(ref qual) = t.qualifier {
    show_type_qualifier(f, &qual);
    let _ = f.write_str(" ");
  }

  show_type_specifier(f, state, &t.ty);
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
  if let Some(ref qual) = field.qualifier {
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

  let _ = f.write_str(";\n");
}

pub fn show_array_spec<F>(f: &mut F, a: &syntax::ArraySpecifier) where F: Write {
  match *a {
    syntax::ArraySpecifier::Unsized => { let _ = f.write_str("[]"); }
    syntax::ArraySpecifier::ExplicitlySized(ref e) => {
      let _ = f.write_str("[");
      show_expr(f, &e);
      let _ = f.write_str("]");
    }
  }
}

pub fn show_arrayed_identifier<F>(f: &mut F, a: &syntax::ArrayedIdentifier) where F: Write {
  let _ = write!(f, "{}", a.ident);

  if let Some(ref arr_spec) = a.array_spec {
    show_array_spec(f, arr_spec);
  }
}

pub fn show_type_qualifier<F>(f: &mut F, q: &syntax::TypeQualifier) where F: Write {
  let mut qualifiers = q.qualifiers.0.iter();
  let first = qualifiers.next().unwrap();

  show_type_qualifier_spec(f, first);

  for qual_spec in qualifiers {
    let _ = f.write_str(" ");
    show_type_qualifier_spec(f, qual_spec)
  }
}

pub fn show_type_qualifier_spec<F>(f: &mut F, q: &syntax::TypeQualifierSpec) where F: Write {
  match *q {
    syntax::TypeQualifierSpec::Storage(ref s) => show_storage_qualifier(f, &s),
    syntax::TypeQualifierSpec::Layout(ref l) => show_layout_qualifier(f, &l),
    syntax::TypeQualifierSpec::Precision(ref p) => show_precision_qualifier(f, &p),
    syntax::TypeQualifierSpec::Interpolation(ref i) => show_interpolation_qualifier(f, &i),
    syntax::TypeQualifierSpec::Invariant => { let _ = f.write_str("invariant"); },
    syntax::TypeQualifierSpec::Precise => { let _ = f.write_str("precise"); }
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
  match ty {
    hir::Type::Variable(t) => {
      match t.ty.ty {
        syntax::TypeSpecifierNonArray::Float => {
          emit_float(state)
        }
        syntax::TypeSpecifierNonArray::Double => {
          //XXX: actually use double here
          emit_float(state)
        }
        syntax::TypeSpecifierNonArray::Vec4 => {
          emit_vec4(state)
        }
        _ => panic!("{:?}", t.ty.ty)
      }
    }
    _ => panic!()
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
  let float = emit_float(state);
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
          let mut name = state.hir.sym(*sym).name.as_str();
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
    hir::ExprKind::Bracket(ref e, ref a) => {
      show_hir_expr(f, state, &e);
      show_array_spec(f, &a);
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
      show_array_spec(f, &a);
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
    hir::FunIdentifier::Expr(ref e) => show_hir_expr(f, state, &*e)
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
  }
}

pub fn show_function_prototype<F>(f: &mut F, state: &mut OutputState, fp: &hir::FunctionPrototype) where F: Write {
  show_fully_specified_type(f, state, &fp.ty);
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
  show_type_specifier(f, state, &p.ty);
  let _ = f.write_str(" ");
  show_arrayed_identifier(f, &p.ident);
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

  let ty = emit_float(state);
  let output_var = state.builder.variable(ty, None, spirv::StorageClass::Function, None);
  state.emitted_syms.insert(d.name.unwrap(),  Variable{ location: output_var, ty });

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
  show_fully_specified_type(f, state, &d.ty);

  if let Some(ref name) = d.name {
    let _ = f.write_str(" ");
    show_sym(f, state, name);
  }

  if let Some(ref arr_spec) = d.array_specifier {
    show_array_spec(f, arr_spec);
  }

  if let Some(ref initializer) = d.initializer {
    let _ = f.write_str(" = ");
    show_initializer(f, state, initializer);
  }
}

pub fn show_single_declaration_no_type<F>(f: &mut F, state: &mut OutputState, d: &hir::SingleDeclarationNoType) where F: Write {
  show_arrayed_identifier(f, &d.ident);

  if let Some(ref initializer) = d.initializer {
    let _ = f.write_str(" = ");
    show_initializer(f, state, initializer);
  }
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
    show_arrayed_identifier(f, ident);
  }
}

pub fn translate_type(state: &mut OutputState, ty: &syntax::FullySpecifiedType) -> spirv::Word {
  emit_void(state)
}

pub fn translate_function_definition(state: &mut OutputState, fd: &hir::FunctionDefinition) {
  state.return_type = Some(Box::new(fd.prototype.ty.clone()));


  let ret_type = translate_type(state, &fd.prototype.ty);
  {
    let void = emit_void(state);
    let b = &mut state.builder;
    let voidf = b.type_function(void, vec![void]);

    let fun = b.begin_function(ret_type,
                     None,
                     (spirv::FunctionControl::DONT_INLINE |
                         spirv::FunctionControl::CONST),
                     voidf)
        .unwrap();
    b.entry_point(spirv::ExecutionModel::Vertex, fun, "main", []);

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
    hir::Condition::Assignment(ref ty, ref name, ref initializer) => {
      show_fully_specified_type(f, state, ty);
      let _ = f.write_str(" ");
      show_identifier(f, name);
      let _ = f.write_str(" = ");
      show_initializer(f, state, initializer);
    }
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
    hir::ExternalDeclaration::Declaration(ref d) => panic!()
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
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %3 "main"
OpDecorate %13 Location 0
%1 = OpTypeVoid
%2 = OpTypeFunction %1 %1
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
