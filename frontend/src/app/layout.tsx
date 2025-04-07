import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import ThemeRegistry from "@Sniffer/components/ThemeRegistry";
import RootLayoutClient from "@Sniffer/components/RootLayoutClient";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Synthetic Image Detector",
  description: "Upload an image to check if it's GAN-generated or real",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <RootLayoutClient>
          <ThemeRegistry>
            {children}
          </ThemeRegistry>
        </RootLayoutClient>
      </body>
    </html>
  );
}
