import type { Metadata } from "next";
import { Inter, Space_Grotesk } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Satellite Damage Detection | AI-Powered Building Assessment",
  description:
    "Deep learning solution for automatic building damage assessment using Siamese Swin-Transformers and cross-temporal attention for rapid post-disaster response.",
  keywords: [
    "satellite imagery",
    "damage detection",
    "deep learning",
    "computer vision",
    "disaster response",
    "xView2",
    "Swin Transformer",
    "semantic segmentation",
  ],
  authors: [{ name: "Satellite Damage Detection Team" }],
  openGraph: {
    title: "Satellite Damage Detection | AI-Powered Building Assessment",
    description:
      "Deep learning solution for automatic building damage assessment using Siamese Swin-Transformers.",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "Satellite Damage Detection",
    description:
      "AI-powered building damage assessment using satellite imagery",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${spaceGrotesk.variable} scroll-smooth`}
    >
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <meta name="theme-color" content="#030014" />
      </head>
      <body className="bg-[#030014] text-white font-sans antialiased overflow-x-hidden">
        {children}
      </body>
    </html>
  );
}
