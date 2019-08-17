This is a really messy exploration at compiling glsl to spriv.

It currently only supports compiling programs of the form:
```glsl
void main()
{
    float x = 0.1;
    x = x + 0.1;
    gl_FragColor = vec4(x, 0.4, 0.8, 1.0);
}
```

The overall approach is to parse the glsl using https://github.com/phaazon/glsl,
convert it to higher level 'hir' representation that does some rudimentary type
checking and manages a symbol table. We then walk over the 'hir' representation
and build the SPIRV. The translator is currently intermingled with a pretty printer
because of another project that I'm working on but the intent is for that code
to go away.
