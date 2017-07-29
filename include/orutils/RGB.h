class RGBColor {
public:
	RGBColor(unsigned char _r, unsigned char _g, unsigned char _b)
		: r(_r), g(_g), b(_b) {}
	
	int toInt(void) {
		return (r << 16) | (g << 8) | b;
	}

	unsigned char r;
	unsigned char g;
	unsigned char b;
};