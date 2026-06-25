import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { gitConfig } from './shared';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      // EvSys wordmark. Theme-aware: the near-black variant would vanish on a
      // dark nav, so swap to the light variant under `.dark`.
      title: (
        <>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`${basePath}/logo.svg`} alt="EvSys" className="h-5 w-auto dark:hidden" />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`${basePath}/logo-dark.svg`} alt="EvSys" className="hidden h-5 w-auto dark:block" />
        </>
      ),
    },
    links: [
      {
        type: 'button',
        text: 'Join our community',
        url: 'https://join.slack.com/t/evsys-community/shared_invite/zt-41tnfb6vb-qpVnWw59wP5LcViww_2A4Q',
        external: true,
      },
    ],
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
