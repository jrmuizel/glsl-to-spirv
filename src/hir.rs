use std::iter::{FromIterator, once};
use std::ops::{Deref, DerefMut};
use std::collections::HashMap;

use glsl::syntax::{NonEmpty, TypeSpecifier, TypeSpecifierNonArray};
use glsl::syntax;
use glsl::syntax::StructFieldSpecifier;
use glsl::syntax::PrecisionQualifier;
use glsl::syntax::ArrayedIdentifier;
use glsl::syntax::ArraySpecifier;
use glsl::syntax::TypeName;
use glsl::syntax::StructSpecifier;
use glsl::syntax::UnaryOp;
use glsl::syntax::BinaryOp;
use glsl::syntax::AssignmentOp;
use glsl::syntax::Identifier;
use crate::hir::SymDecl::Function;
use crate::hir::Initializer::Simple;
use crate::hir::SimpleStatement::Jump;

trait LiftFrom<S> {
    fn lift(state: &mut State, s: S) -> Self;
}

fn lift<S, T: LiftFrom<S>>(state: &mut State, s: S) -> T {
    LiftFrom::lift(state, s)
}

#[derive(Debug)]
pub struct Symbol {
    pub name: String,
    pub decl: SymDecl
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    ret: Type,
    params: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionType {
    signatures: NonEmpty<FunctionSignature>
}

#[derive(Clone, Debug, PartialEq)]
pub enum StorageClass {
    None,
    Const,
    In,
    Out,
    Uniform,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArraySizes {
    pub sizes: Vec<Expr>
}

impl LiftFrom<&ArraySpecifier> for ArraySizes {
    fn lift(state: &mut State, a: &ArraySpecifier) -> Self {
        ArraySizes{ sizes: vec![match a {
            ArraySpecifier::Unsized=> panic!(),
            ArraySpecifier::ExplicitlySized(expr) => translate_expression(state, expr)
        }]}
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeKind {
    Void,
    Bool,
    Int,
    UInt,
    Float,
    Double,
    Vec2,
    Vec3,
    Vec4,
    DVec2,
    DVec3,
    DVec4,
    BVec2,
    BVec3,
    BVec4,
    IVec2,
    IVec3,
    IVec4,
    UVec2,
    UVec3,
    UVec4,
    Mat2,
    Mat3,
    Mat4,
    Mat23,
    Mat24,
    Mat32,
    Mat34,
    Mat42,
    Mat43,
    DMat2,
    DMat3,
    DMat4,
    DMat23,
    DMat24,
    DMat32,
    DMat34,
    DMat42,
    DMat43,
    // floating point opaque types
    Sampler1D,
    Image1D,
    Sampler2D,
    Image2D,
    Sampler3D,
    Image3D,
    SamplerCube,
    ImageCube,
    Sampler2DRect,
    Image2DRect,
    Sampler1DArray,
    Image1DArray,
    Sampler2DArray,
    Image2DArray,
    SamplerBuffer,
    ImageBuffer,
    Sampler2DMS,
    Image2DMS,
    Sampler2DMSArray,
    Image2DMSArray,
    SamplerCubeArray,
    ImageCubeArray,
    Sampler1DShadow,
    Sampler2DShadow,
    Sampler2DRectShadow,
    Sampler1DArrayShadow,
    Sampler2DArrayShadow,
    SamplerCubeShadow,
    SamplerCubeArrayShadow,
    // signed integer opaque types
    ISampler1D,
    IImage1D,
    ISampler2D,
    IImage2D,
    ISampler3D,
    IImage3D,
    ISamplerCube,
    IImageCube,
    ISampler2DRect,
    IImage2DRect,
    ISampler1DArray,
    IImage1DArray,
    ISampler2DArray,
    IImage2DArray,
    ISamplerBuffer,
    IImageBuffer,
    ISampler2DMS,
    IImage2DMS,
    ISampler2DMSArray,
    IImage2DMSArray,
    ISamplerCubeArray,
    IImageCubeArray,
    // unsigned integer opaque types
    AtomicUInt,
    USampler1D,
    UImage1D,
    USampler2D,
    UImage2D,
    USampler3D,
    UImage3D,
    USamplerCube,
    UImageCube,
    USampler2DRect,
    UImage2DRect,
    USampler1DArray,
    UImage1DArray,
    USampler2DArray,
    UImage2DArray,
    USamplerBuffer,
    UImageBuffer,
    USampler2DMS,
    UImage2DMS,
    USampler2DMSArray,
    UImage2DMSArray,
    USamplerCubeArray,
    UImageCubeArray,
    Struct(SymRef)
}

impl LiftFrom<&syntax::TypeSpecifierNonArray> for TypeKind {
    fn lift(state: &mut State, spec: &syntax::TypeSpecifierNonArray) -> Self {
        use syntax::TypeSpecifierNonArray;
        use TypeKind::*;
        match spec {
            TypeSpecifierNonArray::Void => Void,
            TypeSpecifierNonArray::Bool => Bool,
            TypeSpecifierNonArray::Int => Int,
            TypeSpecifierNonArray::UInt => UInt,
            TypeSpecifierNonArray::Float => Float,
            TypeSpecifierNonArray::Double => Double,
            TypeSpecifierNonArray::Vec2 => Vec2,
            TypeSpecifierNonArray::Vec3 => Vec3,
            TypeSpecifierNonArray::Vec4 => Vec4,
            TypeSpecifierNonArray::DVec2 => DVec2,
            TypeSpecifierNonArray::DVec3 => DVec3,
            TypeSpecifierNonArray::DVec4 => DVec4,
            TypeSpecifierNonArray::BVec2 => BVec2,
            TypeSpecifierNonArray::BVec3 => BVec3,
            TypeSpecifierNonArray::BVec4 => BVec4,
            TypeSpecifierNonArray::IVec2 => IVec2,
            TypeSpecifierNonArray::IVec3 => IVec3,
            TypeSpecifierNonArray::IVec4 => IVec4,
            TypeSpecifierNonArray::UVec2 => UVec2,
            TypeSpecifierNonArray::UVec3 => UVec3,
            TypeSpecifierNonArray::UVec4 => UVec4,
            TypeSpecifierNonArray::Mat2 => Mat2,
            TypeSpecifierNonArray::Mat3 => Mat3,
            TypeSpecifierNonArray::Mat4 => Mat4,
            TypeSpecifierNonArray::Mat23 => Mat23,
            TypeSpecifierNonArray::Mat24 => Mat24,
            TypeSpecifierNonArray::Mat32 => Mat32,
            TypeSpecifierNonArray::Mat34 => Mat34,
            TypeSpecifierNonArray::Mat42 => Mat42,
            TypeSpecifierNonArray::Mat43 => Mat43,
            TypeSpecifierNonArray::DMat2 => DMat2,
            TypeSpecifierNonArray::DMat3 => DMat3,
            TypeSpecifierNonArray::DMat4 => DMat4,
            TypeSpecifierNonArray::DMat23 => DMat23,
            TypeSpecifierNonArray::DMat24 => DMat24,
            TypeSpecifierNonArray::DMat32 => DMat32,
            TypeSpecifierNonArray::DMat34 => DMat34,
            TypeSpecifierNonArray::DMat42 => DMat42,
            TypeSpecifierNonArray::DMat43 => DMat43,
            TypeSpecifierNonArray::Sampler1D => Sampler1D,
            TypeSpecifierNonArray::Image1D => Image1D,
            TypeSpecifierNonArray::Sampler2D => Sampler2D,
            TypeSpecifierNonArray::Image2D => Image2D,
            TypeSpecifierNonArray::Sampler3D => Sampler3D,
            TypeSpecifierNonArray::Image3D => Image3D,
            TypeSpecifierNonArray::SamplerCube => SamplerCube,
            TypeSpecifierNonArray::ImageCube => ImageCube,
            TypeSpecifierNonArray::Sampler2DRect => Sampler2DRect,
            TypeSpecifierNonArray::Image2DRect => Image2DRect,
            TypeSpecifierNonArray::Sampler1DArray => Sampler1DArray,
            TypeSpecifierNonArray::Image1DArray => Image1DArray,
            TypeSpecifierNonArray::Sampler2DArray => Sampler2DArray,
            TypeSpecifierNonArray::Image2DArray => Image2DArray,
            TypeSpecifierNonArray::SamplerBuffer => SamplerBuffer,
            TypeSpecifierNonArray::ImageBuffer => ImageBuffer,
            TypeSpecifierNonArray::Sampler2DMS => Sampler2DMS,
            TypeSpecifierNonArray::Image2DMS => Image2DMS,
            TypeSpecifierNonArray::Sampler2DMSArray => Sampler2DMSArray,
            TypeSpecifierNonArray::Image2DMSArray => Image2DMSArray,
            TypeSpecifierNonArray::SamplerCubeArray => SamplerCubeArray,
            TypeSpecifierNonArray::ImageCubeArray => ImageCubeArray,
            TypeSpecifierNonArray::Sampler1DShadow => Sampler1DShadow,
            TypeSpecifierNonArray::Sampler2DShadow => Sampler2DShadow,
            TypeSpecifierNonArray::Sampler2DRectShadow => Sampler2DRectShadow,
            TypeSpecifierNonArray::Sampler1DArrayShadow => Sampler1DArrayShadow,
            TypeSpecifierNonArray::Sampler2DArrayShadow => Sampler2DArrayShadow,
            TypeSpecifierNonArray::SamplerCubeShadow => SamplerCubeShadow,
            TypeSpecifierNonArray::SamplerCubeArrayShadow => SamplerCubeArrayShadow,
            TypeSpecifierNonArray::ISampler1D => ISampler1D,
            TypeSpecifierNonArray::IImage1D => IImage1D,
            TypeSpecifierNonArray::ISampler2D => ISampler2D,
            TypeSpecifierNonArray::IImage2D => IImage2D,
            TypeSpecifierNonArray::ISampler3D => ISampler3D,
            TypeSpecifierNonArray::IImage3D => IImage3D,
            TypeSpecifierNonArray::ISamplerCube => ISamplerCube,
            TypeSpecifierNonArray::IImageCube => IImageCube,
            TypeSpecifierNonArray::ISampler2DRect => ISampler2DRect,
            TypeSpecifierNonArray::IImage2DRect => IImage2DRect,
            TypeSpecifierNonArray::ISampler1DArray => ISampler1DArray,
            TypeSpecifierNonArray::IImage1DArray => IImage1DArray,
            TypeSpecifierNonArray::ISampler2DArray => ISampler2DArray,
            TypeSpecifierNonArray::IImage2DArray => IImage2DArray,
            TypeSpecifierNonArray::ISamplerBuffer => ISamplerBuffer,
            TypeSpecifierNonArray::IImageBuffer => IImageBuffer,
            TypeSpecifierNonArray::ISampler2DMS => ISampler2DMS,
            TypeSpecifierNonArray::IImage2DMS => IImage2DMS,
            TypeSpecifierNonArray::ISampler2DMSArray => ISampler2DMSArray,
            TypeSpecifierNonArray::IImage2DMSArray => IImage2DMSArray,
            TypeSpecifierNonArray::ISamplerCubeArray => ISamplerCubeArray,
            TypeSpecifierNonArray::IImageCubeArray => IImageCubeArray,
            TypeSpecifierNonArray::AtomicUInt => AtomicUInt,
            TypeSpecifierNonArray::USampler1D => USampler1D,
            TypeSpecifierNonArray::UImage1D => UImage1D,
            TypeSpecifierNonArray::USampler2D => USampler2D,
            TypeSpecifierNonArray::UImage2D => UImage2D,
            TypeSpecifierNonArray::USampler3D => USampler3D,
            TypeSpecifierNonArray::UImage3D => UImage3D,
            TypeSpecifierNonArray::USamplerCube => USamplerCube,
            TypeSpecifierNonArray::UImageCube => UImageCube,
            TypeSpecifierNonArray::USampler2DRect => USampler2DRect,
            TypeSpecifierNonArray::UImage2DRect => UImage2DRect,
            TypeSpecifierNonArray::USampler1DArray => USampler1DArray,
            TypeSpecifierNonArray::UImage1DArray => UImage1DArray,
            TypeSpecifierNonArray::USampler2DArray => USampler2DArray,
            TypeSpecifierNonArray::UImage2DArray => UImage2DArray,
            TypeSpecifierNonArray::USamplerBuffer => USamplerBuffer,
            TypeSpecifierNonArray::UImageBuffer => UImageBuffer,
            TypeSpecifierNonArray::USampler2DMS => USampler2DMS,
            TypeSpecifierNonArray::UImage2DMS => UImage2DMS,
            TypeSpecifierNonArray::USampler2DMSArray => USampler2DMSArray,
            TypeSpecifierNonArray::UImage2DMSArray => UImage2DMSArray,
            TypeSpecifierNonArray::USamplerCubeArray => USamplerCubeArray,
            TypeSpecifierNonArray::UImageCubeArray => UImageCubeArray,
            TypeSpecifierNonArray::Struct(s) => {
                Struct(state.lookup(s.name.as_ref().unwrap().as_str()).unwrap())
            }
            TypeSpecifierNonArray::TypeName(s) => {
                Struct(state.lookup(&s.0).unwrap())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Type {
    pub kind: TypeKind,
    pub precision: Option<PrecisionQualifier>,
    pub array_sizes: Option<Box<ArraySizes>>
}

impl Type {
    pub fn new(kind: TypeKind) -> Self {
        Type { kind, precision: None, array_sizes: None }
    }
}

impl LiftFrom<&syntax::FullySpecifiedType> for Type {
    fn lift(state: &mut State, ty: &syntax::FullySpecifiedType) -> Self {
        let kind = lift(state, &ty.ty.ty);
        let array_sizes = match ty.ty.array_specifier.as_ref() {
            Some(x) => Some(Box::new(lift(state, x))),
            None => None
        };
        let precision = get_precision(ty.qualifier.clone());
        Type {
            kind,
            precision,
            array_sizes
        }
    }
}

impl LiftFrom<&syntax::TypeSpecifier> for Type {
    fn lift(state: &mut State, ty: &syntax::TypeSpecifier) -> Self {
        let kind = lift(state, &ty.ty);
        let array_sizes = ty.array_specifier.as_ref().map(|x| Box::new(lift(state, x)));
        Type {
            kind,
            precision: None,
            array_sizes
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub ty: Type,
    pub name: syntax::Identifier,
}

fn get_precision(qualifiers: Option<syntax::TypeQualifier>) -> Option<PrecisionQualifier>{
    let mut precision = None;
    for qual in qualifiers.iter().flat_map(|x| x.qualifiers.0.iter()) {
        match qual {
            syntax::TypeQualifierSpec::Precision(p) => {
                if precision.is_some() {
                    panic!("Multiple precisions");
                }
                precision = Some(p.clone());
            }
            _ => {}
        }
    }
    precision
}

impl LiftFrom<&StructFieldSpecifier> for StructField {
    fn lift(state: &mut State, f: &StructFieldSpecifier) -> Self {
        let precision = get_precision(f.qualifier.clone());
        let mut ty: Type = lift(state, &f.ty);
        match &f.identifiers.0[..] {
            [ident] => {
                if let Some(a) = &ident.array_spec {
                    ty.array_sizes = Some(Box::new(lift(state, a)));
                }
                StructField{ ty, name: ident.ident.clone() }
            }
            _ => panic!("bad number of identifiers")
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructFields {
    pub fields: Vec<StructField>
}

impl LiftFrom<&StructSpecifier> for StructFields {
    fn lift(state: &mut State, s: &StructSpecifier) -> Self {
        let fields = s.fields.0.iter().map(|field| {
            lift(state, field)
        }).collect();
        Self { fields }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SymDecl {
    Function(FunctionType),
    Variable(StorageClass, Type),
    Struct(StructFields)
}

impl SymDecl {
    fn var(t: TypeKind) -> Self {
        SymDecl::Variable(StorageClass::None, Type::new(t))
    }
}

#[derive(Clone, Debug, PartialEq, Copy, Eq, Hash)]
pub struct SymRef(u32);

#[derive(Debug)]
struct Scope {
    name: String,
    names: HashMap<String, SymRef>,
}
impl Scope {
    fn new(name: String) -> Self {  Scope { name, names: HashMap::new() }}
}

#[derive(Debug)]
pub struct State {
    scopes: Vec<Scope>,
    syms: Vec<Symbol>,
}


impl State {
    pub fn new() -> Self {
        State { scopes: Vec::new(), syms: Vec::new() }
    }

    fn lookup(&self, name: &str) -> Option<SymRef> {
        for s in self.scopes.iter().rev() {
            if let Some(sym) = s.names.get(name) {
                return Some(*sym);
            }
        }
        return None;
    }

    fn declare(&mut self, name: &str, decl: SymDecl) -> SymRef {
        let s = SymRef(self.syms.len() as u32);
        self.syms.push(Symbol{ name: name.into(), decl});
        self.scopes.last_mut().unwrap().names.insert(name.into(), s);
        s
    }

    pub fn sym(&self, sym: SymRef) -> &Symbol {
        &self.syms[sym.0 as usize]
    }

    pub fn lookup_sym_mut(&mut self, name: &str) -> Option<&mut Symbol> {
        self.lookup(name).map(move |x| &mut self.syms[x.0 as usize])
    }

    fn push_scope(&mut self, name: String) {
        self.scopes.push(Scope::new(name));
    }
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

/// A declaration.
#[derive(Clone, Debug, PartialEq)]
pub enum Declaration {
    FunctionPrototype(FunctionPrototype),
    StructDefinition(SymRef),
    InitDeclaratorList(InitDeclaratorList),
    Precision(PrecisionQualifier, TypeSpecifier),
    Block(Block),
    Global(TypeQualifier, Vec<Identifier>)
}

/// A general purpose block, containing fields and possibly a list of declared identifiers. Semantic
/// is given with the storage qualifier.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub qualifier: TypeQualifier,
    pub name: Identifier,
    pub fields: Vec<StructFieldSpecifier>,
    pub identifier: Option<ArrayedIdentifier>
}

/// Function identifier.
#[derive(Clone, Debug, PartialEq)]
pub enum FunIdentifier {
    Identifier(SymRef),
    Constructor(Type)
}

/// Function prototype.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionPrototype {
    pub ty: Type,
    pub name: Identifier,
    pub parameters: Vec<FunctionParameterDeclaration>
}

/// Function parameter declaration.
#[derive(Clone, Debug, PartialEq)]
pub enum FunctionParameterDeclaration {
    Named(Option<TypeQualifier>, FunctionParameterDeclarator),
    Unnamed(Option<TypeQualifier>, TypeSpecifier)
}

/// Function parameter declarator.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionParameterDeclarator {
    pub ty: Type,
    pub ident: Identifier
}

/// Init declarator list.
#[derive(Clone, Debug, PartialEq)]
pub struct InitDeclaratorList {
    // XXX it feels like separating out the type and the names is better than
    // head and tail
    // Also, it might be nice to separate out type definitions from name definitions
    pub head: SingleDeclaration,
    pub tail: Vec<SingleDeclarationNoType>
}

/// Type qualifier.
#[derive(Clone, Debug, PartialEq)]
pub struct TypeQualifier {
    pub qualifiers: NonEmpty<TypeQualifierSpec>
}

fn lift_type_qualifier_for_declaration(state: &mut State, q: &Option<syntax::TypeQualifier>) -> Option<TypeQualifier> {
    q.as_ref().and_then(|x| {
        NonEmpty::from_iter(x.qualifiers.0.iter().flat_map(|x| {
            match x {
                syntax::TypeQualifierSpec::Precision(_) => None,
                syntax::TypeQualifierSpec::Interpolation(i) => Some(TypeQualifierSpec::Interpolation(i.clone())),
                syntax::TypeQualifierSpec::Invariant => Some(TypeQualifierSpec::Invariant),
                syntax::TypeQualifierSpec::Layout(l) => Some(TypeQualifierSpec::Layout(l.clone())),
                syntax::TypeQualifierSpec::Precise => Some(TypeQualifierSpec::Precise),
                syntax::TypeQualifierSpec::Storage(s) => None,
            }
        })).map(|x| TypeQualifier{ qualifiers: x})
    })
}

fn lift_type_qualifier_for_parameter(state: &mut State, q: &Option<syntax::TypeQualifier>) -> Option<TypeQualifier> {
    q.as_ref().and_then(|x| {
        NonEmpty::from_iter(x.qualifiers.0.iter().flat_map(|x| {
            match x {
                syntax::TypeQualifierSpec::Precision(_) => None,
                syntax::TypeQualifierSpec::Interpolation(i) => Some(TypeQualifierSpec::Interpolation(i.clone())),
                syntax::TypeQualifierSpec::Invariant => Some(TypeQualifierSpec::Invariant),
                syntax::TypeQualifierSpec::Layout(l) => Some(TypeQualifierSpec::Layout(l.clone())),
                syntax::TypeQualifierSpec::Precise => Some(TypeQualifierSpec::Precise),
                syntax::TypeQualifierSpec::Storage(s) => {
                    match s {
                        syntax::StorageQualifier::Const => Some(TypeQualifierSpec::Parameter(ParameterQualifier::Const)),
                        syntax::StorageQualifier::In => Some(TypeQualifierSpec::Parameter(ParameterQualifier::In)),
                        syntax::StorageQualifier::Out => Some(TypeQualifierSpec::Parameter(ParameterQualifier::Out)),
                        syntax::StorageQualifier::InOut => Some(TypeQualifierSpec::Parameter(ParameterQualifier::InOut)),
                        _ => panic!("Bad type qualifier for parameter")
                    }
                }

            }
        })).map(|x| TypeQualifier{ qualifiers: x})
    })
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParameterQualifier {
    Const,
    In,
    InOut,
    Out,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MemoryQualifier {
    Coherent,
    Volatile,
    Restrict,
    ReadOnly,
    WriteOnly,
}

/// Type qualifier spec.
#[derive(Clone, Debug, PartialEq)]
pub enum TypeQualifierSpec {
    Layout(syntax::LayoutQualifier),
    Interpolation(syntax::InterpolationQualifier),
    Invariant,
    Parameter(ParameterQualifier),
    Memory(MemoryQualifier),
    Precise
}


/// Single declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct SingleDeclaration {
    pub ty: Type,
    pub ty_def: Option<SymRef>,
    pub qualifier: Option<TypeQualifier>,
    pub name: SymRef,
    pub initializer: Option<Initializer>
}

/// A single declaration with implicit, already-defined type.
#[derive(Clone, Debug, PartialEq)]
pub struct SingleDeclarationNoType {
    pub ident: ArrayedIdentifier,
    pub initializer: Option<Initializer>
}

/// Initializer.
#[derive(Clone, Debug, PartialEq)]
pub enum Initializer {
    Simple(Box<Expr>),
    List(NonEmpty<Initializer>)
}

impl From<Expr> for Initializer {
    fn from(e: Expr) -> Self {
        Initializer::Simple(Box::new(e))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Type
}

#[derive(Clone, Debug, PartialEq)]
pub enum FieldSet {
    Rgba,
    Xyzw,
    Stpq,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwizzleSelector {
    pub field_set: FieldSet,
    pub components: Vec<i8>
}

impl SwizzleSelector {
    fn parse(s: &str) -> Self {
        let mut components = Vec::new();
        let mut field_set = Vec::new();

        for c in s.chars() {
            match c {
                'r' => { components.push(0); field_set.push(FieldSet::Rgba); }
                'x' => { components.push(0); field_set.push(FieldSet::Xyzw); }
                's' => { components.push(0); field_set.push(FieldSet::Stpq); }

                'g' => { components.push(1); field_set.push(FieldSet::Rgba); }
                'y' => { components.push(1); field_set.push(FieldSet::Xyzw); }
                't' => { components.push(1); field_set.push(FieldSet::Stpq); }

                'b' => { components.push(2); field_set.push(FieldSet::Rgba); }
                'z' => { components.push(2); field_set.push(FieldSet::Xyzw); }
                'p' => { components.push(2); field_set.push(FieldSet::Stpq); }

                'a' => { components.push(3); field_set.push(FieldSet::Rgba); }
                'w' => { components.push(3); field_set.push(FieldSet::Xyzw); }
                'q' => { components.push(3); field_set.push(FieldSet::Stpq); }
                _ => panic!("bad selector")
            }
        }

        let first = &field_set[0];
        assert!(field_set.iter().all(|item| item == first));
        assert!(components.len() <= 4);
        SwizzleSelector { field_set: first.clone(), components }
    }

    pub fn to_string(&self) -> String {
        let mut s = String::new();
        let fs = match self.field_set {
            FieldSet::Rgba => ['r','g','b','a'],
            FieldSet::Xyzw => ['x', 'y','z','w'],
            FieldSet::Stpq => ['s','t','p','q'],
        };
        for i in &self.components {
            s.push(fs[*i as usize])
        }
        s
    }
}

/// The most general form of an expression. As you can see if you read the variant list, in GLSL, an
/// assignment is an expression. This is a bit silly but think of an assignment as a statement first
/// then an expression which evaluates to what the statement “returns”.
///
/// An expression is either an assignment or a list (comma) of assignments.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    /// A variable expression, using an identifier.
    Variable(SymRef),
    /// Integral constant expression.
    IntConst(i32),
    /// Unsigned integral constant expression.
    UIntConst(u32),
    /// Boolean constant expression.
    BoolConst(bool),
    /// Single precision floating expression.
    FloatConst(f32),
    /// Double precision floating expression.
    DoubleConst(f64),
    /// A unary expression, gathering a single expression and a unary operator.
    Unary(UnaryOp, Box<Expr>),
    /// A binary expression, gathering two expressions and a binary operator.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// A ternary conditional expression, gathering three expressions.
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    /// An assignment is also an expression. Gathers an expression that defines what to assign to, an
    /// assignment operator and the value to associate with.
    Assignment(Box<Expr>, AssignmentOp, Box<Expr>),
    /// Add an array specifier to an expression.
    Bracket(Box<Expr>, Box<Expr>),
    /// A functional call. It has a function identifier and a list of expressions (arguments).
    FunCall(FunIdentifier, Vec<Expr>),
    /// An expression associated with a field selection (struct).
    Dot(Box<Expr>, Identifier),
    /// An expression associated with a component selection
    SwizzleSelector(Box<Expr>, SwizzleSelector),
    /// Post-incrementation of an expression.
    PostInc(Box<Expr>),
    /// Post-decrementation of an expression.
    PostDec(Box<Expr>),
    /// An expression that contains several, separated with comma.
    Comma(Box<Expr>, Box<Expr>)
}

/*
impl From<i32> for Expr {
    fn from(x: i32) -> Expr {
        ExprKind::IntConst(x)
    }
}

impl From<u32> for Expr {
    fn from(x: u32) -> Expr {
        Expr::UIntConst(x)
    }
}

impl From<bool> for Expr {
    fn from(x: bool) -> Expr {
        Expr::BoolConst(x)
    }
}

impl From<f32> for Expr {
    fn from(x: f32) -> Expr {
        Expr::FloatConst(x)
    }
}

impl From<f64> for Expr {
    fn from(x: f64) -> Expr {
        Expr::DoubleConst(x)
    }
}
*/
/// Starting rule.
#[derive(Clone, Debug, PartialEq)]
pub struct TranslationUnit(pub NonEmpty<ExternalDeclaration>);

impl TranslationUnit {
    /// Construct a translation unit from an iterator.
    ///
    /// # Errors
    ///
    /// `None` if the iterator yields no value.
    pub fn from_iter<I>(iter: I) -> Option<Self> where I: IntoIterator<Item = ExternalDeclaration> {
        NonEmpty::from_iter(iter).map(TranslationUnit)
    }
}

impl Deref for TranslationUnit {
    type Target = NonEmpty<ExternalDeclaration>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TranslationUnit {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for TranslationUnit {
    type IntoIter = <NonEmpty<ExternalDeclaration> as IntoIterator>::IntoIter;
    type Item = ExternalDeclaration;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a TranslationUnit {
    type IntoIter = <&'a NonEmpty<ExternalDeclaration> as IntoIterator>::IntoIter;
    type Item = &'a ExternalDeclaration;

    fn into_iter(self) -> Self::IntoIter {
        (&self.0).into_iter()
    }
}

impl<'a> IntoIterator for &'a mut TranslationUnit {
    type IntoIter = <&'a mut NonEmpty<ExternalDeclaration> as IntoIterator>::IntoIter;
    type Item = &'a mut ExternalDeclaration;

    fn into_iter(self) -> Self::IntoIter {
        (&mut self.0).into_iter()
    }
}

/// External declaration.
#[derive(Clone, Debug, PartialEq)]
pub enum ExternalDeclaration {
    Preprocessor(Preprocessor),
    FunctionDefinition(FunctionDefinition),
    Declaration(Declaration)
}

/// Function definition.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionDefinition {
    pub prototype: FunctionPrototype,
    pub statement: CompoundStatement,
}

/// Compound statement (with no new scope).
#[derive(Clone, Debug, PartialEq)]
pub struct CompoundStatement {
    pub statement_list: Vec<Statement>
}

impl FromIterator<Statement> for CompoundStatement {
    fn from_iter<T>(iter: T) -> Self where T: IntoIterator<Item = Statement> {
        CompoundStatement {
            statement_list: iter.into_iter().collect()
        }
    }
}

/// Statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Compound(Box<CompoundStatement>),
    Simple(Box<SimpleStatement>)
}

/// Simple statement.
#[derive(Clone, Debug, PartialEq)]
pub enum SimpleStatement {
    Declaration(Declaration),
    Expression(ExprStatement),
    Selection(SelectionStatement),
    Switch(SwitchStatement),
    Iteration(IterationStatement),
    Jump(JumpStatement)
}

impl SimpleStatement {
    /// Create a new expression statement.
    pub fn new_expr<E>(expr: E) -> Self where E: Into<Expr> {
        SimpleStatement::Expression(Some(expr.into()))
    }

    /// Create a new selection statement (if / else).
    pub fn new_if_else<If, True, False>(
        ife: If,
        truee: True,
        falsee: False
    ) -> Self
        where If: Into<Expr>,
              True: Into<Statement>,
              False: Into<Statement> {
        SimpleStatement::Selection(
            SelectionStatement {
                cond: Box::new(ife.into()),
                rest: SelectionRestStatement::Else(Box::new(truee.into()), Box::new(falsee.into()))
            }
        )
    }

    /// Create a new while statement.
    pub fn new_while<C, S>(
        cond: C,
        body: S
    ) -> Self
        where C: Into<Condition>,
              S: Into<Statement> {
        SimpleStatement::Iteration(
            IterationStatement::While(cond.into(), Box::new(body.into()))
        )
    }

    /// Create a new do-while statement.
    pub fn new_do_while<C, S>(
        body: S,
        cond: C
    ) -> Self
        where S: Into<Statement>,
              C: Into<Expr> {
        SimpleStatement::Iteration(
            IterationStatement::DoWhile(Box::new(body.into()), Box::new(cond.into()))
        )
    }
}

/// Expression statement.
pub type ExprStatement = Option<Expr>;

/// Selection statement.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionStatement {
    pub cond: Box<Expr>,
    pub rest: SelectionRestStatement
}

/// Condition.
#[derive(Clone, Debug, PartialEq)]
pub enum Condition {
    Expr(Box<Expr>),
}

impl From<Expr> for Condition {
    fn from(expr: Expr) -> Self {
        Condition::Expr(Box::new(expr))
    }
}

/// Selection rest statement.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectionRestStatement {
    /// Body of the if.
    Statement(Box<Statement>),
    /// The first argument is the body of the if, the rest is the next statement.
    Else(Box<Statement>, Box<Statement>)
}

/// Switch statement.
#[derive(Clone, Debug, PartialEq)]
pub struct SwitchStatement {
    pub head: Box<Expr>,
    pub cases: Vec<Case>
}

/// Case label statement.
#[derive(Clone, Debug, PartialEq)]
pub enum CaseLabel {
    Case(Box<Expr>),
    Def
}

/// An individual case
#[derive(Clone, Debug, PartialEq)]
pub struct Case {
    pub label: CaseLabel,
    pub stmts: Vec<Statement>
}

/// Iteration statement.
#[derive(Clone, Debug, PartialEq)]
pub enum IterationStatement {
    While(Condition, Box<Statement>),
    DoWhile(Box<Statement>, Box<Expr>),
    For(ForInitStatement, ForRestStatement, Box<Statement>)
}

/// For init statement.
#[derive(Clone, Debug, PartialEq)]
pub enum ForInitStatement {
    Expression(Option<Expr>),
    Declaration(Box<Declaration>)
}

/// For init statement.
#[derive(Clone, Debug, PartialEq)]
pub struct ForRestStatement {
    pub condition: Option<Condition>,
    pub post_expr: Option<Box<Expr>>
}

/// Jump statement.
#[derive(Clone, Debug, PartialEq)]
pub enum JumpStatement {
    Continue,
    Break,
    Return(Box<Expr>),
    Discard
}

/// Some basic preprocessor commands.
///
/// As it’s important to carry them around the AST because they cannot be substituted in a normal
/// preprocessor (they’re used by GPU’s compilers), those preprocessor commands are available for
/// inspection.
///
/// > Important note: so far, only `#version` and `#extension` are supported. Other pragmas will be
/// > added in the future. Stay tuned.
#[derive(Clone, Debug, PartialEq)]
pub enum Preprocessor {
    Define(PreprocessorDefine),
    Version(PreprocessorVersion),
    Extension(PreprocessorExtension)
}

/// A #define preprocessor command.
/// Allows any expression but only Integer and Float literals make sense
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessorDefine {
    pub name: Identifier,
    pub value: Expr,
}

/// A #version preprocessor command.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessorVersion {
    pub version: u16,
    pub profile: Option<PreprocessorVersionProfile>
}

/// A #version profile annotation.
#[derive(Clone, Debug, PartialEq)]
pub enum PreprocessorVersionProfile {
    Core,
    Compatibility,
    ES
}

/// An #extension preprocessor command.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessorExtension {
    pub name: PreprocessorExtensionName,
    pub behavior: Option<PreprocessorExtensionBehavior>
}

/// An #extension name annotation.
#[derive(Clone, Debug, PartialEq)]
pub enum PreprocessorExtensionName {
    /// All extensions you could ever imagine in your whole lifetime (how crazy is that!).
    All,
    /// A specific extension.
    Specific(String)
}

/// An #extension behavior annotation.
#[derive(Clone, Debug, PartialEq)]
pub enum PreprocessorExtensionBehavior {
    Require,
    Enable,
    Warn,
    Disable
}

trait NonEmptyExt<T> {
    fn map<U, F: FnMut(&mut State, &T) -> U>(&self, s: &mut State, f: F) -> NonEmpty<U>;
    fn new(x: T) -> NonEmpty<T>;
}

impl<T> NonEmptyExt<T> for NonEmpty<T> {
    fn map<U, F: FnMut(&mut State, &T) -> U>(&self, s: &mut State, mut f: F) -> NonEmpty<U> {
        NonEmpty::from_iter(self.into_iter().map(|x| f(s, &x))).unwrap()
    }
    fn new(x: T) -> NonEmpty<T> {
        NonEmpty::from_iter(vec![x].into_iter()).unwrap()
    }
}


fn translate_initializater(state: &mut State, i: &syntax::Initializer) -> Initializer {
    match i {
        syntax::Initializer::Simple(i) => Initializer::Simple(Box::new(translate_expression(state, i))),
        _ => panic!()
    }
}

fn translate_struct_declaration(state: &mut State, d: &syntax::SingleDeclaration) -> Declaration {
    let mut ty = d.ty.clone();
    let ty_def = match &ty.ty.ty {
        TypeSpecifierNonArray::Struct(s) => {
            let decl = SymDecl::Struct(lift(state, s));
            Some(state.declare(s.name.as_ref().unwrap().as_str(), decl))
        }
        _ => None
    };

    let ty_def = ty_def.expect("Must be type definition");

    Declaration::StructDefinition(ty_def)
}

fn translate_single_declaration(state: &mut State, d: &syntax::SingleDeclaration) -> SingleDeclaration {
    let mut ty = d.ty.clone();
    ty.ty.array_specifier = d.array_specifier.clone();
    let ty_def = match &ty.ty.ty {
        TypeSpecifierNonArray::Struct(s) => {
            let decl = SymDecl::Struct(lift(state, s));
            Some(state.declare(s.name.as_ref().unwrap().as_str(), decl))
        }
        _ => None
    };

    let mut ty: Type = lift(state, &d.ty);
    if let Some(array) = &d.array_specifier {
        ty.array_sizes = Some(Box::new(lift(state, array)))
    }

    let name = match d.name.as_ref() {
        Some(name) => {
            let mut storage = StorageClass::None;
            for qual in d.ty.qualifier.iter().flat_map(|x| x.qualifiers.0.iter()) {
                match qual {
                    syntax::TypeQualifierSpec::Storage(s) => {
                        match (&storage, s) {
                            (StorageClass::None, syntax::StorageQualifier::Out) => {
                                storage = StorageClass::Out
                            }
                            (StorageClass::None, syntax::StorageQualifier::In) => {
                                storage = StorageClass::In
                            }
                            (StorageClass::None, syntax::StorageQualifier::Uniform) => {
                                storage = StorageClass::Uniform
                            }
                            (StorageClass::None, syntax::StorageQualifier::Const) => {
                                storage = StorageClass::Const
                            }
                            _ => panic!("bad storage {:?}", (storage, s))
                        }
                    }
                    _ => {}
                }
            }
            let decl = SymDecl::Variable(storage, ty.clone());
            Some(state.declare(d.name.as_ref().unwrap().as_str(), decl))
        }
        None => None
    };



    SingleDeclaration {
        qualifier: lift_type_qualifier_for_declaration(state, &d.ty.qualifier),
        name: name.expect("must have name"),
        ty,
        ty_def,
        initializer: d.initializer.as_ref().map(|x| translate_initializater(state, x)),
    }
}

fn translate_single_declaration_no_type(state: &mut State, d: &syntax::SingleDeclarationNoType) -> SingleDeclarationNoType {
    panic!()
}

fn translate_init_declarator_list(state: &mut State, l: &syntax::InitDeclaratorList) -> Declaration {
    match &l.head.name {
        Some(name) => {
            Declaration::InitDeclaratorList(InitDeclaratorList {
                head: translate_single_declaration(state, &l.head),
                tail: l.tail.iter().map(|x| translate_single_declaration_no_type(state, x)).collect()
            })
        }
        None => {
            translate_struct_declaration(state, &l.head)
        }
    }

}

fn translate_declaration(state: &mut State, d: &syntax::Declaration) -> Declaration {
    match d {
        syntax::Declaration::Block(b) => Declaration::Block(panic!()),
        syntax::Declaration::FunctionPrototype(p) => Declaration::FunctionPrototype(translate_prototype(state, p)),
        syntax::Declaration::Global(ty, ids) => Declaration::Global(panic!(), panic!()),
        syntax::Declaration::InitDeclaratorList(dl) => translate_init_declarator_list(state, dl),
        syntax::Declaration::Precision(p, ts) => Declaration::Precision(p.clone(), ts.clone()),
    }
}

fn is_vector(ty: &Type) -> bool {
    match ty.kind {
        TypeKind::Vec2 | TypeKind::Vec3 | TypeKind::Vec4 |
        TypeKind::IVec2 | TypeKind::IVec3 | TypeKind::IVec4 => {
            ty.array_sizes == None
        }
        _ => false
    }
}

fn index_matrix(ty: &Type) -> Option<TypeKind> {
    use TypeKind::*;
    if ty.array_sizes != None {
        return None
    }
    Some(match ty.kind {
        Mat2 => Vec2,
        Mat3 => Vec3,
        Mat4 => Vec4,
        Mat23 => Vec3,
        Mat24 => Vec4,
        Mat32 => Vec2,
        Mat34 => Vec4,
        Mat42 => Vec2,
        Mat43 => Vec3,
        DMat2 => DVec2,
        DMat3 => DVec3,
        DMat4 => DVec4,
        DMat23 => DVec3,
        DMat24 => DVec4,
        DMat32 => DVec2,
        DMat34 => DVec4,
        DMat42 => DVec2,
        DMat43 => DVec3,
        _ => return None
        })
}



fn is_ivec(ty: &Type) -> bool {
    match ty.kind {
        TypeKind::IVec2 | TypeKind::IVec3 | TypeKind::IVec4 => {
            ty.array_sizes == None
        }
        _ => false
    }
}

fn compatible_type(lhs: &Type, rhs: &Type) -> bool {
    if lhs == &Type::new(TypeKind::Double) &&
        rhs == &Type::new(TypeKind::Float) {
        true
    } else if rhs == &Type::new(TypeKind::Double) &&
        lhs == &Type::new(TypeKind::Float) {
        true
    } else if rhs == &Type::new(TypeKind::Int) &&
        lhs == &Type::new(TypeKind::Float) {
        true
    } else if rhs == &Type::new(TypeKind::Float) &&
        lhs == &Type::new(TypeKind::Int) {
        true
    } else {
        lhs.kind == rhs.kind && lhs.array_sizes == rhs.array_sizes
    }
}

fn promoted_type(lhs: &Type, rhs: &Type) -> Type {
    if lhs == &Type::new(TypeKind::Double) &&
        rhs == &Type::new(TypeKind::Float) {
        Type::new(TypeKind::Double)
    } else if lhs == &Type::new(TypeKind::Float) &&
        rhs == &Type::new(TypeKind::Double) {
        Type::new(TypeKind::Double)
    } else if is_vector(&lhs) && (
        rhs == &Type::new(TypeKind::Float) ||
        rhs == &Type::new(TypeKind::Double) ||
        rhs == &Type::new(TypeKind::Int)
    ) {
        // scalars promote to vectors
        lhs.clone()
    } else if is_vector(&rhs) && (
        lhs == &Type::new(TypeKind::Float) ||
        lhs == &Type::new(TypeKind::Double) ||
        lhs == &Type::new(TypeKind::Int)
    ) {
        // scalars promote to vectors
        rhs.clone()
    } else {
        assert_eq!(lhs, rhs);
        lhs.clone()
    }
}

fn translate_expression(state: &mut State, e: &syntax::Expr) -> Expr {
    match e {
        syntax::Expr::Variable(i) => {
            let sym = match state.lookup(i.as_str()) {
                Some(sym) => sym,
                None => panic!("missing declaration {}", i.as_str())
            };
            let ty = match &state.sym(sym).decl {
                SymDecl::Variable(_, ty) => ty.clone(),
                _ => panic!("bad variable type")
            };
            Expr { kind: ExprKind::Variable(sym), ty }
        },
        syntax::Expr::Assignment(lhs, op, rhs) => {
            let lhs = Box::new(translate_expression(state, lhs));
            let rhs = Box::new(translate_expression(state, rhs));
            if !compatible_type(&lhs.ty, &rhs.ty) {
                panic!("incompatible {:?} {:?}", lhs, rhs)
            }
            let ty = lhs.ty.clone();
            Expr { kind: ExprKind::Assignment(lhs, op.clone(), rhs), ty }
        }
        syntax::Expr::Binary(op, lhs, rhs) => {
            let lhs = Box::new(translate_expression(state, lhs));
            let rhs = Box::new(translate_expression(state, rhs));
            let ty = if op == &BinaryOp::Mult {
                if lhs.ty.kind == TypeKind::Mat3 && rhs.ty.kind == TypeKind::Vec3 {
                    rhs.ty.clone()
                } else if lhs.ty.kind == TypeKind::Mat4 && rhs.ty.kind == TypeKind::Vec4 {
                    rhs.ty.clone()
                } else {
                    promoted_type(&lhs.ty, &rhs.ty)
                }
            } else {
                promoted_type(&lhs.ty, &rhs.ty)
            };

            // comparison operators have a bool result
            let ty = match op {
                BinaryOp::Equal |
                BinaryOp::GT |
                BinaryOp::GTE |
                BinaryOp::LT |
                BinaryOp::LTE => Type::new(TypeKind::Bool),
                _ => ty
            };


            Expr { kind: ExprKind::Binary(op.clone(), lhs, rhs), ty}
        }
        syntax::Expr::Unary(op, e) => {
            let e = Box::new(translate_expression(state, e));
            let ty = e.ty.clone();
            Expr { kind: ExprKind::Unary(op.clone(), e), ty}
        }
        syntax::Expr::BoolConst(b) => {
            Expr { kind: ExprKind::BoolConst(*b), ty: Type::new(TypeKind::Bool) }
        }
        syntax::Expr::Comma(lhs, rhs) => {
            let lhs = Box::new(translate_expression(state, lhs));
            let rhs = Box::new(translate_expression(state, rhs));
            assert_eq!(lhs.ty, rhs.ty);
            let ty = lhs.ty.clone();
            Expr { kind: ExprKind::Comma(lhs, rhs), ty }
        }
        syntax::Expr::DoubleConst(d) => {
            Expr { kind: ExprKind::DoubleConst(*d), ty: Type::new(TypeKind::Double) }
        }
        syntax::Expr::FloatConst(f) => {
            Expr { kind: ExprKind::FloatConst(*f), ty: Type::new(TypeKind::Float) }
        },
        syntax::Expr::FunCall(fun, params) => {
            let ret_ty: Type;
            let params: Vec<Expr> = params.iter().map(|x| translate_expression(state, x)).collect();
            Expr {
                kind:
                ExprKind::FunCall(
                    match fun {
                        syntax::FunIdentifier::Identifier(i) => {
                            let sym = match state.lookup(i.as_str()) {
                                Some(s) => s,
                                None => panic!("missing symbol {}", i.as_str())
                            };
                            match &state.sym(sym).decl {
                                SymDecl::Function(fn_ty) => {
                                    let mut ret = None;
                                    for sig in &fn_ty.signatures {
                                        let mut matching = true;
                                        for (e, p) in params.iter().zip(sig.params.iter()) {
                                            if !compatible_type(&e.ty, p) {
                                                matching = false;
                                                break;
                                            }
                                        }
                                        if matching {
                                            ret = Some(sig.ret.clone());
                                            break;
                                        }
                                    }
                                    ret_ty = match ret {
                                        Some(t) => t,
                                        None => {
                                            dbg!(&fn_ty.signatures);
                                            dbg!(params.iter().map(|p| p).collect::<Vec<_>>());
                                            panic!("no matching func {}", i.as_str())
                                        }
                                    };
                                },
                                SymDecl::Struct(t) => {
                                    ret_ty = Type::new(TypeKind::Struct(sym))
                                }
                                _ => panic!("can only call functions")
                            };

                            FunIdentifier::Identifier(sym)
                        },
                        syntax::FunIdentifier::Expr(e) => {
                            let ty = match &**e {
                                syntax::Expr::Bracket(i, array) => {
                                    let kind = match &**i {
                                        syntax::Expr::Variable(i) => {
                                            match i.as_str() {
                                                "vec4" => TypeKind::Vec4,
                                                _ => panic!()
                                            }
                                        }
                                        _ => panic!()
                                    };

                                    Type { kind, precision: None, array_sizes: Some(Box::new(lift(state, array)))}
                                }
                                _ => panic!()
                            };
                            ret_ty = ty.clone();

                            FunIdentifier::Constructor(ty)

                        }
                    },
                    params
                ),
                ty: ret_ty,
            }
        }
        syntax::Expr::IntConst(i) => {
            Expr { kind: ExprKind::IntConst(*i), ty: Type::new(TypeKind::Int) }
        }
        syntax::Expr::UIntConst(u) => {
            Expr { kind: ExprKind::UIntConst(*u), ty: Type::new(TypeKind::UInt) }
        }
        syntax::Expr::PostDec(e) => {
            let e = Box::new(translate_expression(state, e));
            let ty = e.ty.clone();
            Expr { kind: ExprKind::PostDec(e), ty }
        }
        syntax::Expr::PostInc(e) => {
            let e = Box::new(translate_expression(state, e));
            let ty = e.ty.clone();
            Expr { kind: ExprKind::PostInc(e), ty }
        }
        syntax::Expr::Ternary(cond, lhs, rhs) => {
            let cond = Box::new(translate_expression(state, cond));
            let lhs = Box::new(translate_expression(state, lhs));
            let rhs = Box::new(translate_expression(state, rhs));
            assert_eq!(lhs.ty, rhs.ty);
            let ty = lhs.ty.clone();
            Expr { kind: ExprKind::Ternary(cond, lhs, rhs), ty }
        }
        syntax::Expr::Dot(e, i) => {
            let e = Box::new(translate_expression(state, e));
            let ty = e.ty.clone();
            let ivec = is_ivec(&ty);
            if is_vector(&ty) {
                let ty = Type::new(match i.as_str().len() {
                    1 => if ivec { TypeKind::Int } else { TypeKind::Float },
                    2 => if ivec { TypeKind::IVec2 } else { TypeKind::Vec2 },
                    3 => if ivec { TypeKind::IVec3 } else { TypeKind::Vec3 },
                    4 => if ivec { TypeKind::IVec4 } else { TypeKind::Vec4 },
                    _ => panic!(),
                });

                Expr { kind: ExprKind::SwizzleSelector(e, SwizzleSelector::parse(i.as_str())), ty }
            } else {
                match ty.kind {
                    TypeKind::Struct(s) => {
                        let fields = match &state.sym(s).decl {
                            SymDecl::Struct(fields) => fields,
                            _ => panic!("expected struct"),
                        };
                        let field = fields.fields.iter().find(|x| &x.name == i).expect("missing field");
                        Expr { kind: ExprKind::Dot(e, i.clone()), ty: field.ty.clone() }
                    }
                    _ => panic!("expected struct found {:#?} {:#?}", e, ty)
                }
            }
        }
        syntax::Expr::Bracket(e, specifier) =>{
            let e = Box::new(translate_expression(state, e));
            let ty = if is_vector(&e.ty) {
                Type::new(TypeKind::Float)
            } else if let Some(ty) = index_matrix(&e.ty) {
                Type::new(ty)
            } else {
                let a = match &e.ty.array_sizes {
                    Some(a) => {
                        let mut a = *a.clone();
                        a.sizes.pop();
                        if a.sizes.len() == 0 {
                            None
                        } else {
                            Some(Box::new(a))
                        }
                    },
                    _ => panic!("{:#?}", e)
                };
                Type { kind: e.ty.kind.clone(), precision: e.ty.precision.clone(), array_sizes: a }
            };
            let indx = match specifier {
                ArraySpecifier::Unsized => panic!("need expression"),
                ArraySpecifier::ExplicitlySized(e) => translate_expression(state, e)
            };
            Expr { kind: ExprKind::Bracket(e, Box::new(indx)), ty }
        }
    }
}

fn translate_switch(state: &mut State, s: &syntax::SwitchStatement) -> SwitchStatement {
    let mut cases = Vec::new();

    let mut case = None;
    for stmt in &s.body {
        match stmt {
            syntax::Statement::Simple(s) => {
                match &**s {
                    syntax::SimpleStatement::CaseLabel(label) => {
                        match case.take() {
                            Some(case) => cases.push(case),
                            _ => {}
                        }
                        case = Some(Case { label: translate_case(state, &label), stmts: Vec::new() })
                    }
                    _ => {
                        match case {
                            Some(ref mut case) => case.stmts.push(translate_statement(state, stmt)),
                            _ => panic!("switch must start with case")
                        }
                    }
                }
            }
            _ => {
                match case {
                    Some(ref mut case) => case.stmts.push(translate_statement(state, stmt)),
                    _ => panic!("switch must start with case")
                }
            }
        }

    }
    match case.take() {
        Some(case) => cases.push(case),
        _ => {}
    }
    SwitchStatement {
        head: Box::new(translate_expression(state, &s.head)),
        cases
    }
}

fn translate_jump(state: &mut State, s: &syntax::JumpStatement) -> JumpStatement {
    match s {
        syntax::JumpStatement::Break => JumpStatement::Break,
        syntax::JumpStatement::Continue => JumpStatement::Continue,
        syntax::JumpStatement::Discard => JumpStatement::Discard,
        syntax::JumpStatement::Return(e) => JumpStatement::Return(Box::new(translate_expression(state, e)))
    }
}

fn translate_condition(state: &mut State, c: &syntax::Condition) -> Condition {
    match c {
        syntax::Condition::Expr(e) => Condition::Expr(Box::new(translate_expression(state, e))),
        _ => panic!()
    }
}

fn translate_for_init(state: &mut State, s: &syntax::ForInitStatement) -> ForInitStatement {
    match s {
        syntax::ForInitStatement::Expression(e) => ForInitStatement::Expression(e.as_ref().map(|e| translate_expression(state, e))),
        syntax::ForInitStatement::Declaration(d) => ForInitStatement::Declaration(Box::new(translate_declaration(state, d))),
    }
}

fn translate_for_rest(state: &mut State, s: &syntax::ForRestStatement) -> ForRestStatement {
    ForRestStatement {
        condition: s.condition.as_ref().map(|c| translate_condition(state, c)),
        post_expr: s.post_expr.as_ref().map(|e| Box::new(translate_expression(state, e)))
    }
}

fn translate_iteration(state: &mut State, s: &syntax::IterationStatement) -> IterationStatement {
    match s {
        syntax::IterationStatement::While(cond, s) =>
            IterationStatement::While(translate_condition(state, cond), Box::new(translate_statement(state, s))),
        syntax::IterationStatement::For(init, rest, s) =>
            IterationStatement::For(translate_for_init(state, init),translate_for_rest(state, rest), Box::new(translate_statement(state, s))),
        syntax::IterationStatement::DoWhile(s, e) =>
            IterationStatement::DoWhile(Box::new(translate_statement(state, s)), Box::new(translate_expression(state, e))),
    }
}

fn translate_case(state: &mut State, c: &syntax::CaseLabel) -> CaseLabel {
    match c {
        syntax::CaseLabel::Def => CaseLabel::Def,
        syntax::CaseLabel::Case(e) => CaseLabel::Case(Box::new(translate_expression(state, e)))
    }
}

fn translate_selection_rest(state: &mut State, s: &syntax::SelectionRestStatement) -> SelectionRestStatement {
    match s {
        syntax::SelectionRestStatement::Statement(s) => SelectionRestStatement::Statement(Box::new(translate_statement(state, s))),
        syntax::SelectionRestStatement::Else(if_body, rest) => {
            SelectionRestStatement::Else(Box::new(translate_statement(state, if_body)), Box::new(translate_statement(state, rest)))
        }
    }
}

fn translate_selection(state: &mut State, s: &syntax::SelectionStatement) -> SelectionStatement {
    SelectionStatement {
        cond: Box::new(translate_expression(state, &s.cond)),
        rest: translate_selection_rest(state, &s.rest),
    }
}

fn translate_simple_statement(state: &mut State, s: &syntax::SimpleStatement) -> SimpleStatement {
    match s {
        syntax::SimpleStatement::Declaration(d) => SimpleStatement::Declaration(translate_declaration(state, d)),
        syntax::SimpleStatement::Expression(e) => SimpleStatement::Expression(e.as_ref().map(|e| translate_expression(state, e))),
        syntax::SimpleStatement::Iteration(i) => SimpleStatement::Iteration(translate_iteration(state, i)),
        syntax::SimpleStatement::Selection(s) => SimpleStatement::Selection(translate_selection(state, s)),
        syntax::SimpleStatement::Jump(j) => SimpleStatement::Jump(translate_jump(state, j)),
        syntax::SimpleStatement::Switch(s) => SimpleStatement::Switch(translate_switch(state, s)),
        syntax::SimpleStatement::CaseLabel(s) => panic!("should be handled by translate_switch")
    }
}

fn translate_statement(state: &mut State, s: &syntax::Statement) -> Statement {
    match s {
        syntax::Statement::Compound(s) => Statement::Compound(Box::new(translate_compound_statement(state, s))),
        syntax::Statement::Simple(s) => Statement::Simple(Box::new(translate_simple_statement(state, s)))
    }
}


fn translate_compound_statement(state: &mut State, cs: &syntax::CompoundStatement) -> CompoundStatement {
    CompoundStatement { statement_list: cs.statement_list.iter().map(|x| translate_statement(state, x)).collect() }
}

fn translate_function_parameter_declarator(state: &mut State, d: &syntax::FunctionParameterDeclarator) -> FunctionParameterDeclarator {
    let mut ty: Type = lift(state, &d.ty);
    if let Some(a) = &d.ident.array_spec {
        ty.array_sizes = Some(Box::new(lift(state, a)));
    }
    FunctionParameterDeclarator {
        ty,
        ident: d.ident.ident.clone(),
    }
}

fn translate_function_parameter_declaration(state: &mut State, p: &syntax::FunctionParameterDeclaration) ->
  FunctionParameterDeclaration
{
    match p {
        syntax::FunctionParameterDeclaration::Named(qual, p) => {
            let decl = SymDecl::Variable(
                StorageClass::None,
                lift(state, &p.ty)

                /*syntax::FullySpecifiedType {
                    qualifier: None,
                    ty: TypeSpecifier {
                        ty: p.ty.ty.clone(),
                        array_specifier: None
                    }
                }*/);
            state.declare(p.ident.ident.as_str(), decl);
            FunctionParameterDeclaration::Named(lift_type_qualifier_for_parameter(state, qual), translate_function_parameter_declarator(state, p))
        }
        syntax::FunctionParameterDeclaration::Unnamed(qual, p) => {
            FunctionParameterDeclaration::Unnamed(lift_type_qualifier_for_parameter(state, qual), p.clone())
        }

    }
}

fn translate_prototype(state: &mut State, cs: &syntax::FunctionPrototype) -> FunctionPrototype {
    FunctionPrototype {
        ty: lift(state, &cs.ty),
        name: cs.name.clone(),
        parameters: cs.parameters.iter().map(|x| translate_function_parameter_declaration(state, x)).collect(),
    }
}

fn translate_function_definition(state: &mut State, fd: &syntax::FunctionDefinition) -> FunctionDefinition {
    let prototype = translate_prototype(state, &fd.prototype);
    let params = prototype.parameters.iter().flat_map(|p| match p {
        FunctionParameterDeclaration::Named(_, p) => Some(p.ty.clone()),
        FunctionParameterDeclaration::Unnamed(_, p) => match p.ty {
            TypeSpecifierNonArray::Void => {
                // just drop void parameters
                None
            },
            _ => panic!() // other unnamed parameters are no good
        },
    }).collect();
    let sig = FunctionSignature{ ret: prototype.ty.clone(), params };
    state.declare(fd.prototype.name.as_str(), SymDecl::Function(FunctionType{ signatures: NonEmpty::new(sig)}));
    state.push_scope(fd.prototype.name.as_str().into());
    let f = FunctionDefinition {
        prototype,
        statement: translate_compound_statement(state, &fd.statement)
    };
    state.pop_scope();
    f
}

fn translate_external_declaration(state: &mut State, ed: &syntax::ExternalDeclaration) -> ExternalDeclaration {
    match ed {
        syntax::ExternalDeclaration::Declaration(d) =>
            ExternalDeclaration::Declaration(translate_declaration(state, d)),
        syntax::ExternalDeclaration::FunctionDefinition(fd) =>
            ExternalDeclaration::FunctionDefinition(translate_function_definition(state, fd)),
        syntax::ExternalDeclaration::Preprocessor(p) =>
            ExternalDeclaration::Preprocessor(panic!())
    }
}

fn declare_function(state: &mut State, name: &str, ret: Type, params: Vec<Type>) {
    let sig = FunctionSignature{ ret, params };
    match state.lookup_sym_mut(name) {
        Some(Symbol { decl: SymDecl::Function(f), ..}) => f.signatures.push(sig),
        None => { state.declare(name, SymDecl::Function(FunctionType{ signatures: NonEmpty::new(sig)})); },
        _ => panic!("overloaded function name {}", name)
    }
    //state.declare(name, Type::Function(FunctionType{ v}))
}


pub fn ast_to_hir(state: &mut State, tu: &syntax::TranslationUnit) -> TranslationUnit {
    // global scope
    state.push_scope("global".into());
    use TypeKind::*;
    declare_function(state, "vec3", Type::new(Vec3),
                     vec![Type::new(Float), Type::new(Float), Type::new(Float)]);
    declare_function(state, "vec3", Type::new(Vec3),
                     vec![Type::new(Float)]);
    declare_function(state, "vec3", Type::new(Vec3),
                     vec![Type::new(Vec2), Type::new(Float)]);
    declare_function(state, "vec4", Type::new(Vec4),
                     vec![Type::new(Vec3), Type::new(Float)]);
    declare_function(state, "vec4", Type::new(Vec4),
                     vec![Type::new(Float), Type::new(Float), Type::new(Float), Type::new(Float)]);
    declare_function(state, "vec4", Type::new(Vec4),
                     vec![Type::new(Vec2), Type::new(Float), Type::new(Float)]);
    declare_function(state, "vec4", Type::new(Vec4),
                     vec![Type::new(Vec2), Type::new(Vec2)]);
    declare_function(state, "vec2", Type::new(Vec2),
                     vec![Type::new(Float)]);
    declare_function(state, "mat3", Type::new(Mat3),
                     vec![Type::new(Vec3), Type::new(Vec3), Type::new(Vec3)]);
    declare_function(state, "mat3", Type::new(Mat3),
                     vec![Type::new(Mat4)]);
    declare_function(state, "abs", Type::new(Float),
                     vec![Type::new(Float), Type::new(Float)]);
    declare_function(state, "dot", Type::new(Float),
                     vec![Type::new(Vec3), Type::new(Vec3)]);
    declare_function(state, "min", Type::new(Vec2),
                     vec![Type::new(Vec2), Type::new(Vec2)]);

    declare_function(state, "mix", Type::new(Vec2),
                     vec![Type::new(Vec2), Type::new(Vec2), Type::new(Vec2)]);
    declare_function(state, "mix", Type::new(Vec3),
                     vec![Type::new(Vec3), Type::new(Vec3), Type::new(Vec3)]);
    declare_function(state, "mix", Type::new(Vec4),
                     vec![Type::new(Vec4), Type::new(Vec4), Type::new(Vec4)]);
    declare_function(state, "mix", Type::new(Vec4),
                     vec![Type::new(Vec4), Type::new(Vec4), Type::new(Float)]);
    declare_function(state, "mix", Type::new(Vec3),
                     vec![Type::new(Vec3), Type::new(Vec3), Type::new(Float)]);
    declare_function(state, "mix", Type::new(Float),
                     vec![Type::new(Float), Type::new(Float), Type::new(Float)]);

    declare_function(state, "step", Type::new(Vec2),
                     vec![Type::new(Vec2), Type::new(Vec2)]);
    declare_function(state, "max", Type::new(Vec2),
                     vec![Type::new(Vec2), Type::new(Vec2)]);
    declare_function(state, "max", Type::new(Float),
                     vec![Type::new(Float), Type::new(Float)]);
    declare_function(state, "min", Type::new(Float),
                     vec![Type::new(Float), Type::new(Float)]);
    declare_function(state, "fwidth", Type::new(Vec2),
                     vec![Type::new(Vec2)]);
    declare_function(state, "clamp", Type::new(Vec3),
                     vec![Type::new(Vec3), Type::new(Float), Type::new(Float)]);
    declare_function(state, "clamp", Type::new(Double),
                     vec![Type::new(Double), Type::new(Double), Type::new(Double)]);
    declare_function(state, "clamp", Type::new(Vec2),
                     vec![Type::new(Vec2), Type::new(Vec2), Type::new(Vec2)]);
    declare_function(state, "clamp", Type::new(Vec3),
                     vec![Type::new(Vec3), Type::new(Vec3), Type::new(Vec3)]);
    declare_function(state, "length", Type::new(Float), vec![Type::new(Vec2)]);
    declare_function(state, "pow", Type::new(Vec3), vec![Type::new(Vec3)]);
    declare_function(state, "pow", Type::new(Float), vec![Type::new(Float)]);
    declare_function(state, "lessThanEqual", Type::new(BVec3),
                     vec![Type::new(Vec3), Type::new(Vec3)]);
    declare_function(state, "if_then_else", Type::new(Vec3),
                     vec![Type::new(BVec3), Type::new(Vec3), Type::new(Vec3)]);
    declare_function(state, "floor", Type::new(Vec4),
                     vec![Type::new(Vec4)]);
    declare_function(state, "floor", Type::new(Double),
                     vec![Type::new(Double)]);
    declare_function(state, "int", Type::new(Int),
                     vec![Type::new(Float)]);
    declare_function(state, "float", Type::new(Float),
                     vec![Type::new(Float)]);
    declare_function(state, "int", Type::new(Int),
                     vec![Type::new(UInt)]);
    declare_function(state, "uint", Type::new(UInt),
                     vec![Type::new(Float)]);
    declare_function(state, "uint", Type::new(UInt),
                     vec![Type::new(Int)]);
    declare_function(state, "ivec2", Type::new(IVec2),
                     vec![Type::new(UInt), Type::new(UInt)]);
    declare_function(state, "ivec2", Type::new(IVec2),
                     vec![Type::new(Int), Type::new(Int)]);
    declare_function(state, "ivec4", Type::new(IVec4),
                     vec![Type::new(Int), Type::new(Int), Type::new(Int), Type::new(Int)]);
    declare_function(state, "texelFetch", Type::new(Vec4),
                     vec![Type::new(Sampler2D), Type::new(IVec2), Type::new(Int)]);
    declare_function(state, "texelFetch", Type::new(IVec4),
                     vec![Type::new(ISampler2D), Type::new(IVec2), Type::new(Int)]);
    declare_function(state, "texture", Type::new(Vec4),
                     vec![Type::new(Sampler2D), Type::new(Vec3)]);
    declare_function(state, "texture", Type::new(Vec4),
                     vec![Type::new(Sampler2D), Type::new(Vec2)]);
    declare_function(state, "transpose", Type::new(Mat3),
                     vec![Type::new(Mat3)]);
    state.declare("gl_FragCoord", SymDecl::var(Vec4));
    state.declare("gl_FragColor", SymDecl::var(Vec4));
    state.declare("gl_Position", SymDecl::var(Vec4));


    TranslationUnit(tu.0.map(state, translate_external_declaration))
}
