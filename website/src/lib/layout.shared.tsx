import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { appName, gitConfig } from './shared';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      // JSX supported — logo + wordmark.
      title: (
        <>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`${basePath}/logo.png`}
            alt=""
            width={28}
            height={28}
            style={{ borderRadius: 6 }}
          />
          {appName}
        </>
      ),
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
