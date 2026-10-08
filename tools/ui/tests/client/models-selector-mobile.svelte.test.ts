// Guards the phone model picker: the dropdown serves a phone a bottom drawer
// with the same search, sections and manage entry the desktop menu shows.

import ModelsSelectorDropdownWrapper from './components/ModelsSelectorDropdownWrapper.svelte';
import { ServerRole } from '$lib/enums';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import type { ModelOption } from '$lib/types/models';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const PHONE = { height: 844, width: 390 };
const DESKTOP = { height: 900, width: 1280 };
const option: ModelOption = {
	capabilities: [],
	id: 'org/Qwen3-8B:Q4_K_M',
	model: 'org/Qwen3-8B:Q4_K_M',
	name: 'Qwen3-8B'
};
const other: ModelOption = {
	capabilities: [],
	id: 'org/Qwen3-8B:Q8_0',
	model: 'org/Qwen3-8B:Q8_0',
	name: 'Qwen3-8B-Q8_0'
};

describe('model picker drawer on a phone', () => {
	beforeEach(async () => {
		serverStore.role = ServerRole.ROUTER;
		// the models read through the providers data layer on this layer of the stack
		modelsStore.activeModels = [option, other];
		await page.viewport(PHONE.width, PHONE.height);
	});

	afterEach(async () => {
		modelsStore.activeModels = [];
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('opens the picker in a drawer', async () => {
		const screen = render(ModelsSelectorDropdownWrapper, { currentModel: option.model });

		await screen.getByRole('button').click();

		await expect.element(screen.getByPlaceholder('Search models...')).toBeVisible();
		await expect.element(screen.getByText('Manage models')).toBeVisible();
	});
});
