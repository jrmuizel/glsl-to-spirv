This is a really messy exploration at compiling glsl to spriv.

It currently only supports compiling programs of the form:
```glsl
void main()
{
    gl_FragColor = vec4(0.2, 0.4, 0.8, 1.0);
}
```
